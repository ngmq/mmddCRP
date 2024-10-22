#include "mmddCRP.h"
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/range/algorithm.hpp>
#include <vector>
#include <iostream>

mmddCRP::mmddCRP(const Eigen::MatrixXd &data, double C, double lambda, double alpha, double gamma, double S, unsigned int seed)
    : ca_(data.rows()),
      C_(C),
      lambda_(lambda),
      alpha_(alpha),
      gamma_(gamma),
      S_(S),
      data_(data),      
      tables_(data),
      rng_( seed )
{
    std::cout << "C_ = " << C_ << std::endl;
    std::cout << "lambda_ = " << lambda_ << std::endl;
    std::cout << "alpha_ = " << alpha_ << std::endl;
    std::cout << "gamma_ = " << gamma_ << std::endl;
    std::cout << "S_ = " << S_ << std::endl;

    //std::cout << data_.minCoeff() << ", " << data_.maxCoeff() << std::endl ;

    colwiseMin_ = data_.colwise().minCoeff();
    colwiseMax_ = data_.colwise().maxCoeff();

    double _min = data_.minCoeff();
    double _max = data_.maxCoeff();

    // normalize the features to be in a same range [min, max] so that the all features
    // are considered equally when calculating the Euclidean distances
    // x -> t = (x - xmin) / (xmax - xmin) in [0, 1] => x = xmin + t * (xmax - xmin)
    // t -> k = _min + t * (_max - _min) = _min + (x - xmin) / (xmax - xmin) * (_max - _min)

    for(std::size_t i = 0; i < data_.rows(); ++i)
    {
        for(std::size_t j = 0; j < data_.cols(); ++j)
        {
            data_(i, j) = 2.0 * (data_(i, j) - colwiseMin_(j)) / (colwiseMax_(j) - colwiseMin_(j)) - 1.0;
        }
    }

    initmean_ = data_.colwise().mean();
    data_.rowwise() -= initmean_.transpose();

    pairwiseDistance_ = Eigen::MatrixXd::Zero(data_.rows(), data_.rows());
    Eigen::VectorXd norm_ = Eigen::MatrixXd::Zero(1, data_.rows()).row(0);
    for(std::size_t i = 0; i < data_.rows(); ++i)
    {
        norm_(i) = data_.row(i).norm();
    }
    for(std::size_t i = 0; i < data_.rows(); ++i)
    {
        for(std::size_t j = i; j < data_.rows(); ++j)
        {
            /// Euclidean distance:
            // pairwiseDistance_(i, j) = (data_.row(i) - data_.row(j)).norm();
            /// Cosine distance:
            pairwiseDistance_(i, j) = std::acos( (data_.row(i).dot(data_.row(j))) / (norm_(i) * norm_(j)) );
            pairwiseDistance_(j, i) = pairwiseDistance_(i, j);
        }
    }
    
    tables_ = Eigen::MatrixXd::Random(data.rows(), data.cols());

    std::cout << "Shape of initmean_: " << initmean_.rows() << ", " << initmean_.cols() << std::endl;
    std::cout << "Shape of tables_:" << tables_.rows() << ", " << tables_.cols() << std::endl;

    data_indices.resize(data_.rows());
    for(std::size_t i = 0; i < data.rows(); ++i) data_indices[i] = i;
}

void mmddCRP::iterate(bool debug)
{
    double min_p_link = std::numeric_limits<double>::max(), max_p_link = -std::numeric_limits<double>::max();

    std::vector<double> p_link;
    boost::range::random_shuffle(data_indices);
    for ( std::size_t isource = 0; isource < num_customers(); ++isource)
    {
        std::size_t source = data_indices[isource];

        ca_.unlink(source);
        ca_.remove_empty_tables();
        create_empty_table(source);

        get_link_likelihoods(source, p_link);
        //if(debug && source == 3)
        //{
        // std::cout << "============= source = " << source << std::endl;
        // std::cout << "*** p_link.size = " << p_link.size() << std::endl;
        for(std::size_t i = 0; i < p_link.size(); ++i)
        {
            //std::cout << "p_link[" << i << "] = " << p_link[i] << std::endl;
            if( min_p_link > p_link[i] ) min_p_link = p_link[i];
            if( max_p_link < p_link[i] ) max_p_link = p_link[i];

            p_link[i] = std::exp(p_link[i]);
        }
        //}
        //std::transform(p_link.begin(), p_link.end(), p_link.begin(), [](double p){return std::exp(p); } );
        boost::random::discrete_distribution<std::size_t> d(p_link.begin(), p_link.end());
        std::size_t _new_label = d(rng_);
        
        //std::cout << "sampled = " << _new_label << std::endl;
        
        _new_label = ca_.get_real_idx(_new_label);

        update_svm(source, _new_label);
        ca_.link(source, _new_label);

        ca_.remove_empty_tables();
    }
    //std::cout << "ars....\n";
    // update alpha by Adaptive Rejection Sampling
    double k = (double)(num_tables());
    double n = (double)(num_customers());
    //std::cout << "k = " << k << "; n = " << n << std::endl;

    // with alpha = 6.9, n = 100 -> log_alpha_ = log(6.9/100) = -0.8/0.3 = -2.6667

    alpha_ = ars(k, n);
    
    //std::cout << "next alpha = " << alpha_ << "; log(alpha/n) = " << std::log(alpha_/n) << std::endl;
   // std::cout << "min_p_link = " << min_p_link << std::endl << "max_p_link = " << max_p_link << std::endl;
}

void mmddCRP::update_svm(std::size_t source, std::size_t _new_label)
{
    double max_val = -std::numeric_limits<double>::max();
    double score = 0;
    std::size_t max_val_idx = 0;
    for(std::size_t i = 0; i < num_tables(); ++i)
    {
        std::size_t target = ca_.get_real_idx(i);
        double tmp = data_.row(source).dot(tables_.row(target));
        if(target == _new_label)
        {
            score = tmp;
        }
        else
        {
            if(max_val < tmp)
            {
                max_val = tmp;
                max_val_idx = target;
            }
        }
    }
    double margin = score - max_val;
    double loss = 0;
    if(margin < 1)
    {
        loss = 1 - margin;
        double learning_rate = std::min( C_, 0.5 * loss / data_.row(source).squaredNorm() );
        tables_.row(_new_label) += learning_rate * data_.row(source);
        tables_.row(max_val_idx) -= learning_rate * data_.row(source);
    }    
}

void mmddCRP::create_empty_table(std::size_t source)
{
    std::size_t _empty_table_idx = ca_.create_empty_table();    
    tables_.row(_empty_table_idx) =  Eigen::MatrixXd::Random(1, data_.cols()).row(0);
}

void mmddCRP::get_link_likelihoods(std::size_t source, std::vector<double>& p) const
{
    p.resize(num_tables());
    for (std::size_t target = 0; target < p.size(); ++target)
    {
        p[target] = get_link_likelihood(source, target);            
    }
}

double mmddCRP::get_link_likelihood(std::size_t source, std::size_t target) const
{
    double log_link_prior = get_log_link_prior(source, target);
    double log_data_likelihood = get_log_data_likelihood(source, target);
    double p = log_link_prior + log_data_likelihood; 
    //if(source == 3)
    {
        //std::cout << "target = " << target << std::endl;
        //std::cout << "prior = " << log_link_prior << ", ll = " << log_data_likelihood << "; p = " << p << std::endl;
    }
    return p;
}

double mmddCRP::get_log_link_prior(std::size_t source, std::size_t target) const
{
    // CRP: ca_.get_table_size(target)
    // mmddCRP: distance(data[source], table[target])

    if(target != num_tables() - 1)
    {
        target = ca_.get_real_idx(target);
        
        double dist = std::numeric_limits<double>::max(), tmp;

	std::set<std::size_t> ss = ca_.get_table_members(target);
	for(std::size_t customerId : ss)
	{
		tmp = pairwiseDistance_(source, customerId);
		dist = std::min(dist, tmp);
	}
    //return std::log(1.0 * ca_.get_table_size(target) / num_customers());
    return -dist/gamma_;

    //dist = (data_.row(source) - tables_.row(target)).norm();
    //if(source == 3)
    //{
        //std::cout << ca_.get_table_size(target) << ", " << 
        //std::log(1.0 * ca_.get_table_size(target) / num_customers()) << ", " << -dist / gamma_ << std::endl;
    //}
        
    }
    else
    {
        return std::log(1.0 * alpha_ / num_customers());
        //return std::log(0.5 / 1000 * alpha_ / num_customers());
    }
}

double mmddCRP::get_log_data_likelihood(std::size_t source, std::size_t target) const
{
    // x * theta - lambda * ||theta||
    target = ca_.get_real_idx(target);
    return S_ * data_.row(source).dot(tables_.row(target)) - lambda_ * tables_.row(target).squaredNorm();
}

void mmddCRP::print_tables(std::ostream& os) const
{    
    ca_.print_tables(os);
}

void mmddCRP::print_table_vectors() const
{
    std::cout << "------ tables ------\n";
    for(std::size_t i = 0; i < num_tables(); ++i)
    {
        std::cout << tables_.row(ca_.get_real_idx(i)) << std::endl;
    }
    std::cout << "------ initmean_ -------\n";
    std::cout << initmean_ << std::endl;
}

std::size_t mmddCRP::get_table(std::size_t customer) const
{
    return ca_.get_table(customer);
}

std::size_t mmddCRP::num_tables() const
{
    return ca_.num_tables();
}

std::size_t mmddCRP::num_customers() const
{
    return ca_.num_customers();
}

double h(double alpha, double k, double n, double offset)
{
    return -offset + (k - 3.0 / 2.0) * std::log(alpha) - 1.0 / (2.0 * alpha) + lgamma(alpha) - lgamma(alpha + n);
}

double hprime(double alpha, double k, double n)
{
    return (k - 3.0 / 2.0) / alpha + 1.0 / (2.0 * alpha * alpha) + psi(alpha) - psi(alpha + n);
}

double yof(double h, double hprime, double x0, double x)
{
    return h + (x - x0) * hprime;
}

double exp_integral(double h, double hprime, double x0, double c1, double c2)
{
    //std::cout << "yof c2 = " << yof(h, hprime, x0, c2) << std::endl;
    //std::cout << "yof c1 = " << yof(h, hprime, x0, c1) << std::endl;

    return ( std::exp(yof(h, hprime, x0, c2)) - std::exp(yof(h, hprime, x0, c1)) ) / hprime;
}

double mmddCRP::ars(double k, double n)
{
    // note that this is a simplified version of what has been implemeted in matlab
    // knowing in advance that there are only two x_i's at the beginning
    // support = [x0 inf]
    // x1.................................x2...............................................x3=inf
    // z0.............z1...................................................................z2=inf
    double x1 = 2.0 / (n - k + 3.0/2.0); // = deriv_down
    double x2 = k * n / (n - k + 1.0);   // = deriv_up

    //std::cout << "x1 = " << x1 << "; x2 = " << x2 << std::endl;
    //std::cout << "k = " << k << "; n = " << n << std::endl;
    //std::cout << "Adding x1 and x2..............\n";
    double hx1 = h(x1, k, n, 0), hprimex1 = hprime(x1, k, n);
    double hx2 = h(x2, k, n, 0), hprimex2 = hprime(x2, k, n);
    double offset = std::max(hx1, hx2);
    hx1 -= offset;
    hx2 -= offset;

    assert(hprimex1 > 0);
    assert(hprimex2 < 0);
    std::vector<double> X;
    X.push_back(x1);
    X.push_back(x2);

    double u1, u2, xstar, u_k_xstar, exp_u_k_xstar, hxstar;
    // u1 for choosing xstar
    // u2 for deciding to reject/accept xstar

    double z0 = x1;
    double zend = std::numeric_limits<double>::max();
    std::vector<double> Z;
    std::vector<double> cumulativeSum;
    double Normalization;

    for(;;)
    {
        //std::cout << "Keep sampling.......\n";

        std::size_t K = X.size();

        //std::cout << "K = " << K << std::endl;

        std::sort(X.begin(), X.end());
        // for(std::size_t i = 0; i < K; ++i)
        // {
        //     std::cout << "X[" << i << "] = " << X[i] << std::endl;
        // }
        Z.resize(K + 1);
        Z[0] = z0;
        Z[K] = zend;

        //std::cout << "Calculating Z...\n";
        for(std::size_t i = 0; i < K-1; ++i)
        {
            // j = i + 1
            double hi = h(X[i], k, n, offset), hprimei = hprime(X[i], k, n);
            double hj = h(X[i+1], k, n, offset), hprimej = hprime(X[i+1], k, n);
            Z[i + 1] = (hj - hi - X[i+1] * hprimej + X[i] * hprimei) / (hprimei - hprimej);
        }

        //std::cout << "Calculating cumulativeSum...\n";

        cumulativeSum.resize(K + 1);
        cumulativeSum[0] = 0;
        for(std::size_t i = 1; i <= K; ++i)
        {
            double hi = h(X[i-1], k, n, offset), hprimei = hprime(X[i-1], k, n);
            double add = exp_integral(hi, hprimei, X[i-1], Z[i-1], Z[i]);
            cumulativeSum[i] = cumulativeSum[i-1] + add;
            //std::cout << "i = " << i << "; Z[i-1] = " << Z[i-1] << "; Z[i] = " << Z[i]
            //<< "; added = " << add << "; cumulativeSum = " << cumulativeSum[i] << std::endl;
        }
        Normalization = cumulativeSum[K];

        //std::cout << "Normalization = " << Normalization << std::endl;

        // for(std::size_t i = 0; i <= K; ++i)
        // {
        //     std::cout << Z[i] << "    ";
        // }
        // std::cout << "\n";
        // std::cout << "Normalization = " << Normalization << std::endl;
        // for(std::size_t i = 0; i <= K; ++i)
        // {
        //     std::cout << cumulativeSum[i] << "    ";
        // }
        // std::cout << "\n";

        // randomly generate u1, u2 in range (0,1)
        boost::uniform_real<> uni_dist(0,1);
        boost::variate_generator<boost::random::mt19937&, boost::uniform_real<> > uni(rng_, uni_dist);
        u1 = uni();
        u2 = uni();

        xstar = -1;
        for(std::size_t i = K-1; i >= 0; --i)
        {
            if(Normalization * u1 > cumulativeSum[i])
            {
                // xstar is between Z[i] and Z[i+1]
                double hi = h(X[i], k, n, offset), hprimei = hprime(X[i], k, n);
                exp_u_k_xstar = (Normalization * u1 - cumulativeSum[i]) * hprimei + std::exp(yof(hi, hprimei, X[i], Z[i]));
                u_k_xstar = std::log(exp_u_k_xstar);
                xstar = X[i] + (-hi + u_k_xstar) / hprimei;
                break;
            }
        }
        assert(xstar > 0);
        //std::cout << "u1 = " << u1 << "; xstar = " << xstar << std::endl;
        hxstar = h(xstar, k, n, offset);

        //std::cout << "u2 = " << u2 << "; hxstar = " << hxstar << "; u_k_xstar = " << u_k_xstar << std::endl;
        if(u2 < std::exp(hxstar - u_k_xstar))
        {
            // accept xstar!
            //std::cout << "accepted\n";
            break;
        }
        else
        {
            // update hull by inserting xstar into X
            //std::cout << "Accept xstar = " << xstar << std::endl;
            //std::cout << "rejected\n";
            X.push_back(xstar);
        }
    }


    return xstar;
}
