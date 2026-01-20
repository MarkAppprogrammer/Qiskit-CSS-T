#ifndef RELAY_BP_DECODER_H
#define RELAY_BP_DECODER_H

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>

// Simple CSR Matrix structure
struct SparseMatrix {
    int rows;
    int cols;
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> data;

    SparseMatrix(int r, int c) : rows(r), cols(c) {
        indptr.push_back(0);
    }
};

class MinSumBPDecoder {
public:
    int num_checks;
    int num_vars;
    int max_iter;
    double alpha;
    
    // Priors and Posteriors (Log Likelihood Ratios)
    std::vector<double> log_prior_ratios;
    std::vector<double> posterior_ratios;
    std::vector<double> memory_strengths;
    std::vector<int> decoding;

    // Graph structures
    // For Check-to-Variable update: Iterate over checks (rows of H)
    // We store messages q_{vc} in a structure aligned with H's rows
    std::vector<int> h_csr_indptr;
    std::vector<int> h_csr_indices;
    std::vector<double> q_messages; // Variable to Check messages

    // For Variable-to-Check update: Iterate over variables (cols of H)
    // We store messages r_{cv} in a structure aligned with H's cols (H transpose)
    std::vector<int> h_csc_indptr;
    std::vector<int> h_csc_indices;
    std::vector<double> r_messages; // Check to Variable messages
    
    // Mappings to sync messages between the two structures
    // map_q_to_r[k] = index in r_messages corresponding to the same edge as q_messages[k]
    // map_r_to_q[k] = index in q_messages corresponding to the same edge as r_messages[k]
    std::vector<int> map_q_to_r;
    std::vector<int> map_r_to_q;

    int current_iteration;
    bool has_memory;

    MinSumBPDecoder(const std::vector<int>& row_ptr, 
                    const std::vector<int>& col_ind, 
                    int n_rows, int n_cols,
                    const std::vector<double>& priors,
                    int max_iter, double alpha);

    void initialize_decoder();
    void reset_for_new_leg();
    void set_memory_strengths(const std::vector<double>& gammas);
    
    void compute_check_to_variable(const std::vector<int>& syndrome);
    void compute_variable_to_check();
    
    // Returns true if converged
    bool step(const std::vector<int>& syndrome);
    
    // Full decode
    bool decode(const std::vector<int>& syndrome);
    
    std::vector<int> get_hard_decision();

private:
    void build_maps(const std::vector<int>& csc_indptr, const std::vector<int>& csc_indices);
};

class RelayDecoder {
public:
    MinSumBPDecoder bp_decoder;
    int pre_iter;
    int num_legs;
    int iter_per_leg;
    double gamma_min;
    double gamma_max;
    double gamma0;
    
    std::mt19937 rng;

    RelayDecoder(const std::vector<int>& row_ptr, 
                 const std::vector<int>& col_ind, 
                 int n_rows, int n_cols,
                 const std::vector<double>& priors,
                 int pre_iter = 80,
                 int num_legs = 300,
                 int iter_per_leg = 60,
                 double gamma_min = -0.24,
                 double gamma_max = 0.66,
                 double gamma0 = 0.1,
                 unsigned long seed = 42);

    bool decode(const std::vector<int>& syndrome, std::vector<int>& result);

private:
    std::vector<double> generate_gammas();
};

#endif // RELAY_BP_DECODER_H
