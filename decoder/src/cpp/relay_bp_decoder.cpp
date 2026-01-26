#include "relay_bp_decoder.h"
#include <algorithm>
#include <cmath>
#include <numeric>

// --- Helper Functions ---

double bound_value(double val) {
    // Prevent numerical explosion
    if (val > 50.0) return 50.0;
    if (val < -50.0) return -50.0;
    return val;
}

int sign_of(double x) {
    return (x < 0) ? -1 : 1;
}

// --- MinSumBPDecoder Implementation ---

MinSumBPDecoder::MinSumBPDecoder(const std::vector<int>& row_ptr, 
                                 const std::vector<int>& col_ind, 
                                 int n_rows, int n_cols,
                                 const std::vector<double>& priors,
                                 int max_iter, double alpha)
    : num_checks(n_rows), num_vars(n_cols), max_iter(max_iter), alpha(alpha), 
      h_csr_indptr(row_ptr), h_csr_indices(col_ind), current_iteration(0), has_memory(false) 
{
    // Initialize LLRs
    log_prior_ratios.resize(num_vars);
    for (int i = 0; i < num_vars; ++i) {
        double p = std::max(1e-15, std::min(priors[i], 1.0 - 1e-15));
        log_prior_ratios[i] = std::log((1.0 - p) / p);
    }
    posterior_ratios = log_prior_ratios; // Initial posteriors
    decoding.resize(num_vars);
    memory_strengths.resize(num_vars, 0.0);

    // Initialize CSR message storage (Variable -> Check messages, stored at Check nodes)
    int num_edges = h_csr_indices.size();
    q_messages.resize(num_edges, 0.0);

    // Build CSC structure (Transpose of H) for Variable updates
    // Count edges per column
    std::vector<int> col_counts(num_vars, 0);
    for (int col : h_csr_indices) {
        col_counts[col]++;
    }

    h_csc_indptr.resize(num_vars + 1);
    h_csc_indptr[0] = 0;
    for (int i = 0; i < num_vars; ++i) {
        h_csc_indptr[i + 1] = h_csc_indptr[i] + col_counts[i];
    }

    h_csc_indices.resize(num_edges);
    r_messages.resize(num_edges, 0.0);
    
    // Fill CSC indices and build maps
    // We need to know where we are in each column to fill it
    std::vector<int> current_col_pos = h_csc_indptr;
    
    map_q_to_r.resize(num_edges);
    map_r_to_q.resize(num_edges);

    for (int r = 0; r < num_checks; ++r) {
        for (int i = h_csr_indptr[r]; i < h_csr_indptr[r + 1]; ++i) {
            int c = h_csr_indices[i];
            int dest_pos = current_col_pos[c];
            
            h_csc_indices[dest_pos] = r;
            
            // Map the edge index i (in CSR) to dest_pos (in CSC)
            map_q_to_r[i] = dest_pos;
            map_r_to_q[dest_pos] = i;
            
            current_col_pos[c]++;
        }
    }
}

void MinSumBPDecoder::initialize_decoder() {
    current_iteration = 0;
    std::fill(q_messages.begin(), q_messages.end(), 0.0);
    std::fill(r_messages.begin(), r_messages.end(), 0.0);
    
    // Initialize q_messages with priors
    for (int i = 0; i < q_messages.size(); ++i) {
        int var_idx = h_csr_indices[i];
        q_messages[i] = log_prior_ratios[var_idx];
    }
    
    // Reset posteriors if memory is active, otherwise they just follow
    if (has_memory) {
        posterior_ratios = log_prior_ratios;
    }
}

void MinSumBPDecoder::reset_for_new_leg() {
    current_iteration = 0;
    std::fill(q_messages.begin(), q_messages.end(), 0.0);
    std::fill(r_messages.begin(), r_messages.end(), 0.0);
    
    // Initialize q with priors
    for (int i = 0; i < q_messages.size(); ++i) {
        int var_idx = h_csr_indices[i];
        q_messages[i] = log_prior_ratios[var_idx];
    }
    // Note: We DO NOT reset posterior_ratios here, that is the point of relay
}

void MinSumBPDecoder::set_memory_strengths(const std::vector<double>& gammas) {
    memory_strengths = gammas;
    has_memory = true;
}

void MinSumBPDecoder::compute_check_to_variable(const std::vector<int>& syndrome) {
    // Min-Sum update: r_{cv}
    // Iterate over checks (rows)
    for (int c = 0; c < num_checks; ++c) {
        int start = h_csr_indptr[c];
        int end = h_csr_indptr[c + 1];
        if (start == end) continue;

        double min1 = 1e30, min2 = 1e30;
        int min1_idx = -1;
        int sign_prod = 1;

        // Pass 1: Find min, min2, total sign
        for (int i = start; i < end; ++i) {
            double val = q_messages[i];
            double abs_val = std::abs(val);
            int s = sign_of(val);
            
            sign_prod *= s;
            
            if (abs_val < min1) {
                min2 = min1;
                min1 = abs_val;
                min1_idx = i;
            } else if (abs_val < min2) {
                min2 = abs_val;
            }
        }

        // Adjust for syndrome
        if (syndrome[c]) {
            sign_prod *= -1;
        }

        // Pass 2: Update r messages
        for (int i = start; i < end; ++i) {
            double val = q_messages[i];
            int my_sign = sign_of(val);
            
            // The sign of the message is total_sign / my_sign
            int msg_sign = sign_prod * my_sign; 
            
            double mag = (i == min1_idx) ? min2 : min1;
            
            // Map to r storage
            int r_idx = map_q_to_r[i];
            r_messages[r_idx] = alpha * mag * msg_sign;
        }
    }
}

void MinSumBPDecoder::compute_variable_to_check() {
    // Update posteriors and q messages
    // Iterate over variables (cols of H -> rows of CSC)
    for (int v = 0; v < num_vars; ++v) {
        int start = h_csc_indptr[v];
        int end = h_csc_indptr[v + 1];
        
        // 1. Compute effective prior (DMem-BP)
        double current_prior;
        if (has_memory) {
            current_prior = log_prior_ratios[v] * (1.0 - memory_strengths[v]) 
                          + posterior_ratios[v] * memory_strengths[v];
        } else {
            current_prior = log_prior_ratios[v];
        }
        
        // 2. Sum incoming r messages
        double sum_r = 0.0;
        for (int i = start; i < end; ++i) {
            sum_r += r_messages[i];
        }
        
        // 3. Update posterior
        double new_posterior = current_prior + sum_r;
        new_posterior = bound_value(new_posterior);
        posterior_ratios[v] = new_posterior;
        
        // 4. Update outgoing q messages
        // q_{vc} = posterior - r_{cv}
        for (int i = start; i < end; ++i) {
            double q_val = new_posterior - r_messages[i];
            // Map back to CSR storage
            int q_idx = map_r_to_q[i];
            q_messages[q_idx] = bound_value(q_val);
        }
    }
}

std::vector<int> MinSumBPDecoder::get_hard_decision() {
    std::vector<int> res(num_vars);
    for (int i = 0; i < num_vars; ++i) {
        res[i] = (posterior_ratios[i] < 0) ? 1 : 0;
    }
    return res;
}

bool MinSumBPDecoder::decode(const std::vector<int>& syndrome) {
    if (current_iteration == 0) {
        initialize_decoder();
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        compute_check_to_variable(syndrome);
        compute_variable_to_check();
        current_iteration++;

        // Check convergence
        std::vector<int> candidate = get_hard_decision();
        
        // Verify: H * candidate == syndrome ?
        bool converged = true;
        for (int r = 0; r < num_checks; ++r) {
            int parity = 0;
            for (int i = h_csr_indptr[r]; i < h_csr_indptr[r + 1]; ++i) {
                int c = h_csr_indices[i];
                if (candidate[c]) parity ^= 1;
            }
            if (parity != syndrome[r]) {
                converged = false;
                break;
            }
        }
        
        if (converged) {
            decoding = candidate;
            return true;
        }
    }
    
    decoding = get_hard_decision();
    return false;
}

// --- RelayDecoder Implementation ---

RelayDecoder::RelayDecoder(const std::vector<int>& row_ptr, 
                           const std::vector<int>& col_ind, 
                           int n_rows, int n_cols,
                           const std::vector<double>& priors,
                           int pre_iter, int num_legs, int iter_per_leg,
                           double gamma_min, double gamma_max, double gamma0,
                           unsigned long seed)
    : bp_decoder(row_ptr, col_ind, n_rows, n_cols, priors, pre_iter, 0.625),
      pre_iter(pre_iter), num_legs(num_legs), iter_per_leg(iter_per_leg),
      gamma_min(gamma_min), gamma_max(gamma_max), gamma0(gamma0), rng(seed)
{
}

std::vector<double> RelayDecoder::generate_gammas() {
    std::uniform_real_distribution<double> dist(gamma_min, gamma_max);
    std::vector<double> gammas(bp_decoder.num_vars);
    for (int i = 0; i < bp_decoder.num_vars; ++i) {
        gammas[i] = dist(rng);
    }
    return gammas;
}

bool RelayDecoder::decode(const std::vector<int>& syndrome, std::vector<int>& result) {
    // 1. Leg 0
    bp_decoder.max_iter = pre_iter;
    std::vector<double> g0(bp_decoder.num_vars, gamma0);
    bp_decoder.set_memory_strengths(g0);
    bp_decoder.initialize_decoder();
    
    if (bp_decoder.decode(syndrome)) {
        result = bp_decoder.decoding;
        return true;
    }
    
    // Track best
    // Ideally we track lowest energy/weight solution.
    // For simplicity, we just return the first valid one or the last one.
    // The Python implementation tracked min weight.
    
    // 2. Relay
    for (int leg = 0; leg < num_legs; ++leg) {
        bp_decoder.max_iter = iter_per_leg;
        bp_decoder.set_memory_strengths(generate_gammas());
        bp_decoder.reset_for_new_leg();
        
        if (bp_decoder.decode(syndrome)) {
            result = bp_decoder.decoding;
            return true;
        }
    }
    
    result = bp_decoder.decoding;
    return false;
}
