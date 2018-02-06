#include "CustomerAssignment.h"
#include <boost/graph/connected_components.hpp>
#include <iostream>

CustomerAssignment::CustomerAssignment(std::size_t num_customers)
    : tables_(num_customers, -1),
      table_counts_(num_customers, 0),
      n_customers_(num_customers)
{
    active_tables_.resize(num_customers);
    for(std::size_t i = 0; i < num_customers; ++i)
    {
        tables_[i] = i;
        active_tables_[i] = i;
        table_counts_[i] = 1;
    }
}

void CustomerAssignment::print_tables(std::ostream& os) const
{
    for (std::size_t t = 0; t < num_tables(); ++t)
    {
        std::set<std::size_t> ss = get_table_members(get_real_idx(t));
        if(ss.size() == 0)
        {
            //std::cout << "idx = " << t << "; real = " << get_real_idx(t) << "; SOMETHING WRONG\n";
        }
        for(auto ssit = ss.begin(); ssit != ss.end();)
        {
            os << *ssit;
            ++ssit;
            if ( ssit != ss.end())
                os << ", ";
        }

        os << std::endl;
    }
}

std::size_t CustomerAssignment::get_real_idx(std::size_t idx) const
{
    return active_tables_[idx];
}

void CustomerAssignment::unlink(std::size_t source)
{
    std::size_t old_table = tables_[source];
    tables_[source] = -1;
    if(old_table >= 0 && old_table < n_customers_)
    {
        table_counts_[old_table] -= 1;    
    }    
}

void CustomerAssignment::link(std::size_t source, std::size_t target)
{
    tables_[source] = target;
    table_counts_[target] += 1;
}

std::size_t CustomerAssignment::create_empty_table()
{
    std::size_t _empty_table_idx = -1;
    for(std::size_t i = 0; i < n_customers_; ++i)
    {
        if(table_counts_[i] == 0)
        {
            _empty_table_idx = i;
            break;
        }
    }
    active_tables_.push_back(_empty_table_idx);
    return _empty_table_idx;
}

void CustomerAssignment::remove_empty_tables()
{
    for(auto it = active_tables_.begin(); it != active_tables_.end();)
    {
        std::size_t j = *it;
        if(table_counts_[j] == 0)
        {
            it = active_tables_.erase(it);
            break;
        }
        else ++it;
    }
}

bool CustomerAssignment::is_in_table(std::size_t customer, std::size_t table) const
{
    return (tables_[customer] == table);
}

std::size_t CustomerAssignment::num_customers() const
{
    return n_customers_;
}

std::set<std::size_t> CustomerAssignment::get_table_members(std::size_t table) const
{
    std::set<std::size_t> v;
    for (std::size_t c = 0; c < num_customers(); ++c)
    {
        if (is_in_table(c, table))
            v.insert(c);
    }
    return v;
}

std::size_t CustomerAssignment::num_tables() const
{
    //return n_tables_;
    return active_tables_.size();
}

std::size_t CustomerAssignment::get_table(std::size_t customer) const
{
    return tables_[customer];
}

std::size_t CustomerAssignment::get_table_size(std::size_t table) const
{
    return get_table_members(table).size();
}