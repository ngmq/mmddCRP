#ifndef CUSTOMERASSIGNMENT_H
#define CUSTOMERASSIGNMENT_H
#include <boost/function.hpp>
#include <set>
//#include <map>

class CustomerAssignment
{
public:
    CustomerAssignment(std::size_t num_customers);
    void print_tables(std::ostream &os) const;
    void unlink(std::size_t source);
    void link(std::size_t source, std::size_t target);
    std::size_t create_empty_table();
    void remove_empty_tables();

    std::size_t get_real_idx(std::size_t idx) const;
    std::size_t num_customers() const;
    std::size_t num_tables() const;
    std::size_t get_table(std::size_t customer) const;
    std::set<std::size_t> get_table_members(std::size_t table) const;
    std::size_t get_table_size(std::size_t table) const;

private:

    bool is_in_table(std::size_t customer, std::size_t table) const;

    std::size_t n_customers_;
    std::vector<std::size_t> tables_; // which table each customer sits at
    std::vector<std::size_t> table_counts_; // number of customers sit at each table
    std::vector<std::size_t> active_tables_; // tables that have at least 01 customers
    //std::size_t n_tables_; // how many tables are there
};

#endif
