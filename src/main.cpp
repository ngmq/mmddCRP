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
            ("alpha,a", po::value<double>(&alpha)->default_value(4),"Log concentration parameters")
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
    
    //std::cout << featuresWithOutGroundTruth << std::endl;

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
