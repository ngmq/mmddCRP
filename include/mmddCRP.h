#ifndef MMDDCRP_H
#define MMDDCRP_H
#include "CustomerAssignment.h"
#include <eigen3/Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <memory>
#include <iostream>

#define lgamma(z) boost::math::lgamma(z)
#define psi(z) boost::math::digamma(z)

class mmddCRP
{
public:
    mmddCRP(const Eigen::MatrixXd& data, double C, double lambda, double alpha, double gamma, double S, unsigned int seed);
    void iterate(bool debug = false);
    void print_tables(std::ostream &os) const;
    void print_table_vectors() const;
    std::size_t get_table(std::size_t customer) const;
    std::size_t num_tables() const;
    std::size_t num_customers() const;

private:
    void get_link_likelihoods(std::size_t source, std::vector<double>& p) const;
    double get_link_likelihood(std::size_t source, std::size_t target) const;
    double get_log_link_prior(std::size_t source, std::size_t target) const;
    double get_log_data_likelihood(std::size_t source, std::size_t target) const;
    void create_empty_table(std::size_t source);
    void update_svm(std::size_t source, std::size_t _new_label);
    double ars(double k, double n);

    CustomerAssignment ca_;
    Eigen::MatrixXd data_;
    Eigen::MatrixXd tables_;
    Eigen::VectorXd initmean_;
    Eigen::VectorXd colwiseMin_, colwiseMax_;
    Eigen::MatrixXd pairwiseDistance_;
    double C_;
    double lambda_;
    double k_;
    double alpha_;
    double gamma_;
    double S_;
    std::vector<std::size_t> data_indices;

    boost::random::mt19937 rng_;
};

#endif
