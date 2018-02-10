// usage:
// ./../bin/ddcrp-gibbs-example -w true -f data.csv -s 0 -i 1
// ./../bin/ddcrp-gibbs-example -w true -f jain.txt -s 0 -i 1
// ./../bin/ddcrp-gibbs-example -w true -f aggregation.txt -s 0 -i 1
// ./../bin/ddcrp-gibbs-example -w true -f flame.txt -s 0 -i 1

#include "mmddCRP.h"
#include <eigen3/Eigen/Dense>
#include <boost/program_options.hpp>
#include <fstream>
#include <iomanip>

namespace utils
{

template<typename M, int StorageType = Eigen::RowMajor>
M load_csv (const std::string& csvfile, const char delim  = ',')
{
    std::ifstream indata;
    indata.open(csvfile);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, delim)) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, StorageType> >(values.data(), rows, values.size()/rows);
}

}
int main(int argc, char* argv[])
{
    namespace po = boost::program_options;

    po::options_description desc("Allowed options");
    unsigned int seed, num_samples, num_burn_in_samples;
    double C, lambda, alpha, gamma, S;

    //("prior-cov-file,S", po::value<std::string>(), "csv file with prior cluster covariance matrix")

    desc.add_options()
            ("help", "produce help message")
            ("log-decay-file,l", po::value<std::string>(), "csv file containing the log decay function values")
            ("feature-file,f", po::value<std::string>(), "csv file containing the feature vectors of the data points")
            ("Scale,S", po::value<double>(&S)->default_value(1.0), "scale between prior and likelihood")
            ("C", po::value<double>(&C)->default_value(0.01), "SVM regularization constant")
            ("prior-mean-file,m", po::value<std::string>(), "csv file with cluster prior mean")
            ("lambda", po::value<double>(&lambda)->default_value(0.03),"Exponential lambda constant")
            ("alpha,a", po::value<double>(&alpha)->default_value(10),"Log concentration parameters")
            ("nsamples,n", po::value<unsigned int>(&num_samples)->default_value(1), "number of samples to draw")
            ("nburn,b", po::value<unsigned int>(&num_burn_in_samples)->default_value(200), "number of burn-in samples for MCMC")
            ("seed,s", po::value<unsigned int>(&seed)->default_value(233), "RNG seed")
            ("draw-from-prior,p", po::bool_switch()->default_value(false), "draw from ddCRP prior (ignore features and likelihood model)")
            ("index-label-at-last-column,i", po::value<bool>(), "Is index label at last column")
            ("gamma,g", po::value<double>(&gamma)->default_value(1.0), "gamma")
            ("wordy,w", po::bool_switch()->default_value(false), "toggle verbose mode with extra output")
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") || (argc == 1))
    {
        std::cout << desc << "\n";
        return 1;
    }

    // Eigen::MatrixXd log_decay;
    // if (vm.count("log-decay-file"))
    // {
    //     std::string d_file = vm["log-decay-file"].as<std::string>();
    //     if ( vm["wordy"].as<bool>() )
    //         std::cout << "Loading log decay file " << d_file << "\n";
    //     log_decay = utils::load_csv<Eigen::MatrixXd>(d_file);
    //     if ( log_decay.rows() != log_decay.cols())
    //     {
    //         std::cout << "Log decay matrix number of rows (" << log_decay.rows() << ") must match number of columns (" << log_decay.cols() <<")!\n";
    //         return 1;
    //     }
    // }
    // else
    // {
    //     std::cout << "Log decay file must be set!\n";
    //     return 1;
    // }

    Eigen::MatrixXd features;
    if (vm.count("feature-file"))
    {
        std::string f_file = vm["feature-file"].as<std::string>();
        if ( vm["wordy"].as<bool>() )
            std::cout << "Loading feature file " << f_file << "\n";
        features = utils::load_csv<Eigen::MatrixXd>(f_file);
        // if ( log_decay.rows() != features.rows())
        // {
        //     std::cout << "Feature matrix number of rows (" << features.rows() << ") must match number of rows in log decay matrix (" << log_decay.rows() <<")!\n";
        //     return 1;
        // }
    }
    else
    {
        std::cout << "Feature file must be set!\n";
        return 1;	
    }


    /*
    * The label is either in the first column or in the last column.
    */
    bool isIndexLabelAtLastColumn = false;
    if(vm.count("index-label-at-last-column"))
    {
    	//std::cout << "vm[index-label-at-last-column] = " << boost::any_cast<std::string>(vm["index-label-at-last-column"].value()) << std::endl;

    	if( vm["index-label-at-last-column"].as<bool>() )
    	{
    		isIndexLabelAtLastColumn = true;
    	}
    	else
    	{
    		isIndexLabelAtLastColumn = false;
    	}
    }
    else
    {
    	std::cout << "index-label-at-last-column must be set!\n";
        return 1;
    }

    //std::cout << features.rows() << ' ' << features.cols() << std::endl;
    Eigen::MatrixXd featuresWithOutGroundTruth;
    std::cout << "isIndexLabelAtLastColumn = " << isIndexLabelAtLastColumn << std::endl;
    if( isIndexLabelAtLastColumn )
    {
    	featuresWithOutGroundTruth = features.leftCols(features.cols() - 1);	
    }
    else
    {
    	featuresWithOutGroundTruth = features.rightCols(features.cols() - 1);
    }

    Eigen::VectorXd colwiseMin_, colwiseMax_, initmean_;
    colwiseMin_ = featuresWithOutGroundTruth.colwise().minCoeff();
    colwiseMax_ = featuresWithOutGroundTruth.colwise().maxCoeff();

    for(std::size_t i = 0; i < featuresWithOutGroundTruth.rows(); ++i)
    {
        for(std::size_t j = 0; j < featuresWithOutGroundTruth.cols(); ++j)
        {
            featuresWithOutGroundTruth(i, j) = 2.0 * (featuresWithOutGroundTruth(i, j) - colwiseMin_(j)) / (colwiseMax_(j) - colwiseMin_(j)) - 1.0;
        }
    }

    initmean_ = featuresWithOutGroundTruth.colwise().mean();
    featuresWithOutGroundTruth.rowwise() -= initmean_.transpose();

    std::cout << "Read data done!\n";

    std::size_t nrow = features.rows(), ncol = features.cols();

    std::cout << "features shape = (" << nrow << ", " << ncol << ")\n";

    // Eigen::VectorXd norm_ = Eigen::MatrixXd::Zero(1, features.rows()).row(0);
    // Eigen::MatrixXd distances = Eigen::MatrixXd::Zero(features.rows(), features.rows());

    // for(std::size_t i = 0; i < features.rows(); ++i)
    // {
    //     norm_(i) = features.row(i).norm();
    // }

    //std::cerr << "passed this\n";

    // double minInClassDistance[10], maxInClassDistance[10], minOutClassDistance[10], maxOutClassDistance[10];
    // double aveInClassDistance[10], aveOutClassDistance[10];
    // unsigned int cntInClassDistance[10], cntOutClassDistance[10];
    // double maxNorm = 0;

    // for(std::size_t i = 0; i < 10; ++i)
    // {
    //     minInClassDistance[i] = maxInClassDistance[i] = minOutClassDistance[i] = maxOutClassDistance[i] = -1;
    //     aveInClassDistance[i] = aveOutClassDistance[i] = 0;
    //     cntInClassDistance[i] = cntOutClassDistance[i] = 0;
    // }

    // for(std::size_t i = 0; i < nrow; ++i)
    // {
    //     for(std::size_t j = i + 1; j < nrow; ++j)
    //     {
    //         double tmp = 0;
            
            /// Euclidean distance
            //tmp = (featuresWithOutGroundTruth.row(i) - featuresWithOutGroundTruth.row(j)).norm();
            
            
            /// Manhattan distance
            // for(std::size_t k = 0; k < featuresWithOutGroundTruth.cols(); ++k)
            // {
            //     tmp += std::abs( featuresWithOutGroundTruth(i, k) - featuresWithOutGroundTruth(j, k) );
            // }
            

            /// Arc-cosine distance
            // tmp = std::acos( (features.row(i).dot(features.row(j))) / (norm_(i) * norm_(j)) );
            //distances(i, j) = tmp;
            //distances(j, i) = tmp;

            // unsigned int label1 = 0, label2 = 0;

            // if( isIndexLabelAtLastColumn )
            // {
            //     label1 = static_cast<unsigned int>(features(i, ncol - 1));
            //     label2 = static_cast<unsigned int>(features(j, ncol - 1));
            // }
            // else
            // {
            //     label1 = static_cast<unsigned int>(features(i, 0));
            //     label2 = static_cast<unsigned int>(features(j, 0));
            // }

            // if(label1 == label2)
            // {
            //     if( minInClassDistance[label1] < 0 || minInClassDistance[label1] > tmp ) minInClassDistance[label1] = tmp;
            //     if( maxInClassDistance[label1] < 0 || maxInClassDistance[label1] < tmp ) maxInClassDistance[label1] = tmp;
            //     aveInClassDistance[label1] += tmp;
            //     cntInClassDistance[label1] += 1;
            // }
            // else
            // {
            //     if( minOutClassDistance[label1] < 0 || minOutClassDistance[label1] > tmp ) minOutClassDistance[label1] = tmp;
            //     if( maxOutClassDistance[label1] < 0 || maxOutClassDistance[label1] < tmp ) maxOutClassDistance[label1] = tmp;

            //     if( minOutClassDistance[label2] < 0 || minOutClassDistance[label2] > tmp ) minOutClassDistance[label2] = tmp;
            //     if( maxOutClassDistance[label2] < 0 || maxOutClassDistance[label2] < tmp ) maxOutClassDistance[label2] = tmp;

            //     aveOutClassDistance[label1] += tmp;
            //     cntOutClassDistance[label1] += 1;
            // }
        //}

        // double tmpNorm = featuresWithOutGroundTruth.row(i).squaredNorm();
        // if( maxNorm < tmpNorm ) maxNorm = tmpNorm;
    //}

    // for(std::size_t i = 0; i < nrow; ++i)
    // {
    //     std::vector<std::pair<double, unsigned int> > vdp;

    //     unsigned int label1 = 0, label2 = 0;
    //     if( isIndexLabelAtLastColumn )
    //         label1 = static_cast<unsigned int>(features(i, ncol - 1));
    //     else
    //         label1 = static_cast<unsigned int>(features(i, 0));

    //     for(std::size_t j = 0; j < nrow; ++j)
    //     {
            
    //         if( isIndexLabelAtLastColumn )
    //         {
    //             label2 = static_cast<unsigned int>(features(j, ncol - 1));
    //         }
    //         else
    //         {
    //             label2 = static_cast<unsigned int>(features(j, 0));
    //         }

    //         if(j != i)
    //             vdp.push_back(std::pair<double, unsigned int>( distances(i, j), label2));
    //     }
    //     std::sort(vdp.begin(), vdp.end());

    //     const unsigned int K = 10;
    //     unsigned int cntRight = 0;

    //     std::cout << "=== data point " << i << "; label = " << label1 << " ===\n";
    //     for(std::size_t k = 0; k < K; ++k)
    //     {
    //         //std::cout << "neigbor in class " << vdp[k].second << " of distance " << vdp[k].first << std::endl;
    //         if( vdp[k].second == label1 )
    //         {
    //             cntRight += 1;
    //         }
    //     }
    //     std::cout << " cntRight = " << cntRight << std::endl;
    // }

    // for(std::size_t i = 0; i < 10; ++i)
    // {
    //     aveInClassDistance[i] /= (1.0 * cntInClassDistance[i]);
    //     aveOutClassDistance[i] /= (1.0 * cntOutClassDistance[i]);

    //     std::cout << "digit = " << i << std::endl
    //     << "minInClassDistance = " << minInClassDistance[i] << std::endl
    //     << "maxInClassDistance = " << maxInClassDistance[i] << std::endl
    //     << "minOutClassDistance = " << minOutClassDistance[i] << std::endl
    //     << "maxOutClassDistance = " << maxOutClassDistance[i] << std::endl
    //     << "aveInClassDistance = " << aveInClassDistance[i] << std::endl
    //     << "aveOutClassDistance = " << aveOutClassDistance[i] << std::endl;
    // }
    // std::cout << "maxNorm = " << maxNorm << std::endl;
    

    mmddCRP clustering(featuresWithOutGroundTruth, C, lambda, alpha, gamma, S, seed);

    if ( vm["wordy"].as<bool>() )
        std::cout << "Starting sampling!\n";

    for ( unsigned int i = 0; i < num_burn_in_samples; ++i)
    {
    	clustering.iterate(true);
        // skip the samples until burn-in is complete
        if ( vm["wordy"].as<bool>() ){
            std::cout << "************ Burn-in-sample #" << i << "; number of tables = " << clustering.num_tables() << std::endl;

        	//clustering.print_table_vectors();
        	//std::cout << "Burn-in-sample #" << i << "; number of tables = " << clustering.num_tables() << std::endl;
        }
    }
    for ( unsigned int i = 0; i < num_samples; ++i)
    {
       
    	clustering.iterate(true);

        if ( vm["wordy"].as<bool>() )
        {
            std::cout << "Sample " << i << "\n";
            std::cout << "number of tables = " << clustering.num_tables() << std::endl;
            //clustering.print_table_vectors();
        }

        std::stringstream st;
        st << "clustering_" << std::setfill('0') << std::setw(4) << i << ".csv";
        std::ofstream table_file(st.str());
        if (table_file.is_open())
        {
            clustering.print_tables( table_file );
        }
        table_file.close();
    }

    return 0;
}
