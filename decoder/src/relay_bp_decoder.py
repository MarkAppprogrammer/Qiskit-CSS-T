import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

class MinSumBPDecoder:
    def __init__(self, check_matrix, error_priors, max_iter=100, alpha=0.625, gamma0=None):
        """
        Args:
            check_matrix (scipy.sparse.spmatrix): Parity check matrix (H).
            error_priors (np.ndarray): Error probability for each qubit/edge.
            max_iter (int): Maximum iterations.
            alpha (float): Scaling factor for Min-Sum (0.0 to 1.0).
            gamma0 (float, optional): Initial memory strength. If None, standard BP.
        """
        self.check_matrix = check_matrix.tocsr() # H matrix
        self.num_checks, self.num_vars = self.check_matrix.shape
        self.max_iter = max_iter
        self.alpha = alpha
        
        # Calculate Log Prior Ratios: log((1-p)/p)
        # Avoid division by zero
        safe_priors = np.clip(error_priors, 1e-15, 1.0 - 1e-15)
        self.log_prior_ratios = np.log((1.0 - safe_priors) / safe_priors)
        
        # Memory strengths
        self.gamma0 = gamma0
        self.memory_strengths = np.zeros(self.num_vars)
        if gamma0 is not None:
            self.memory_strengths[:] = gamma0
            
        # Posteriors (initialized to priors)
        self.posterior_ratios = self.log_prior_ratios.copy()
        self.decoding = np.zeros(self.num_vars, dtype=int)
        
        # Edge storage for message passing
        # We need efficient mapping between check-centric and variable-centric views.
        # H is CSR, so we can iterate checks easily.
        # For variable-centric, we need CSC or a map.
        
        self.rows, self.cols = self.check_matrix.nonzero()
        self.num_edges = len(self.rows)
        
        # Message arrays (flat, aligned with edges)
        # check_to_var_msgs[k] corresponds to edge (rows[k], cols[k])
        # To make this efficient, we sort edges to match CSR (check-centric)
        # and create a mapping for variable-centric operations.
        
        # CSR storage is naturally sorted by row then col.
        # sort_indices() ensures it.
        self.check_matrix.sort_indices()
        self.h_csr = self.check_matrix
        self.h_csc = self.check_matrix.tocsc()
        self.h_csc.sort_indices()
        
        # Messages:
        # q_msgs: Variable to Check messages (same structure as H)
        # r_msgs: Check to Variable messages (same structure as H)
        
        self.q_msgs = self.h_csr.copy()
        self.q_msgs.data = np.zeros(self.num_edges, dtype=np.float64)
        
        self.r_msgs = self.h_csr.copy()
        self.r_msgs.data = np.zeros(self.num_edges, dtype=np.float64)
        
        self.current_iteration = 0

    def initialize_decoder(self):
        self.current_iteration = 0
        self.q_msgs.data[:] = 0.0
        self.r_msgs.data[:] = 0.0
        
        # Initialize variable-to-check messages with priors
        self.q_msgs.data = self.log_prior_ratios[self.q_msgs.indices]
        
        # Reset posteriors if using memory (to priors)
        self.posterior_ratios = self.log_prior_ratios.copy()
        if self.gamma0 is not None:
             self.memory_strengths[:] = self.gamma0

    def reset_for_new_leg(self):
        """
        Resets message graph for a new relay leg, but KEEPS posterior_ratios
        as the 'memory' from the previous leg.
        """
        self.current_iteration = 0
        self.q_msgs.data[:] = 0.0
        self.r_msgs.data[:] = 0.0
        
        # Reset variable messages to priors (standard initialization)
        # The 'memory' effect happens in compute_variable_to_check using posterior_ratios
        self.q_msgs.data = self.log_prior_ratios[self.q_msgs.indices]

    def set_memory_strengths(self, gammas):
        self.memory_strengths = gammas

    def compute_check_to_variable(self, syndrome):
        """
        Min-Sum update: r_{cv} = (-1)^sigma_c * prod_{v' != v} sgn(q_{cv'}) * min_{v' != v} |q_{cv'}|
        """
        # Optimized vectorized implementation
        
        # 1. Prepare data
        q_data = self.q_msgs.data
        q_sign = np.sign(q_data)
        # Avoid zero sign issues, though usually LLRs are non-zero.
        q_sign[q_sign == 0] = 1 
        q_abs = np.abs(q_data)
        
        indptr = self.q_msgs.indptr
        
        # 2. Check update loop
        # We will iterate row by row. 
        # For high performance, this part should ideally be in Cython/Numba, 
        # but pure Python/NumPy is okay for prototyping.
        
        r_data_new = np.zeros_like(q_data)
        
        for i in range(self.num_checks):
            start = indptr[i]
            end = indptr[i+1]
            if start == end:
                continue
            
            # Get messages for this check
            row_q_abs = q_abs[start:end]
            row_q_sign = q_sign[start:end]
            
            # Total sign product
            total_sign = np.prod(row_q_sign)
            
            # Adjust for syndrome
            # Syndrome is 0 or 1. If 1, flip sign.
            syndrome_sign = -1 if syndrome[i] else 1
            total_sign *= syndrome_sign
            
            # Find min and second min absolute values
            if len(row_q_abs) >= 2:
                # We need the two smallest values
                part_idx = np.argpartition(row_q_abs, 1)
                min1_idx = part_idx[0]
                min1 = row_q_abs[min1_idx]
                
                # We need to find the second min accurately. 
                # argpartition does not guarantee order of other elements, 
                # but part_idx[1:] contains them. We take min of those.
                # Actually for small degree (like 6-12), sorting is fast enough or full argpartition.
                # Let's just use part_idx[1] if size is small, but strictly argpartition only guarantees pivot.
                # A safer way for second min:
                remaining = np.delete(row_q_abs, min1_idx)
                min2 = np.min(remaining)
            elif len(row_q_abs) == 1:
                min1 = row_q_abs[0]
                min2 = min1 # Should ideally be huge, but consistent with min-sum approx
                min1_idx = 0
            else:
                continue
                
            # Update r messages
            # For each edge j in this row
            for k in range(len(row_q_abs)):
                # Sign: total_sign * sgn(q_jk)
                # sgn(q_jk) is 1 or -1. dividing is same as multiplying.
                msg_sign = total_sign * row_q_sign[k]
                
                # Magnitude: min1 unless this is the min1, then min2
                msg_mag = min2 if k == min1_idx else min1
                
                # Apply alpha scaling
                val = self.alpha * msg_mag * msg_sign
                r_data_new[start + k] = val
                
        self.r_msgs.data = r_data_new

    def compute_variable_to_check(self):
        """
        Update variable to check messages (q) and posteriors (M).
        q_{vc} = Lambda_v + sum_{c' != c} r_{c'v}
        """
        # Strategy:
        # 1. Calculate full sum for each variable (Posterior M_v)
        # 2. Subtract r_{cv} to get q_{vc}
        
        # Convert r_msgs to CSC to sum over columns efficiently
        r_csc = self.r_msgs.tocsc()
        
        # 1. Compute Priors with Memory
        # Lambda_v(t) = (1 - gamma_v) * Lambda_v(0) + gamma_v * M_v(t-1)
        if self.gamma0 is not None:
            # Vectorized update
            current_priors = (self.log_prior_ratios * (1 - self.memory_strengths) + 
                              self.posterior_ratios * self.memory_strengths)
        else:
            current_priors = self.log_prior_ratios
            
        # 2. Calculate new Posteriors (sum of all incoming r messages + prior)
        # r_csc.sum(axis=0) sums columns. Returns matrix of shape (1, num_vars)
        incoming_sum = np.array(r_csc.sum(axis=0)).flatten()
        new_posteriors = current_priors + incoming_sum
        
        # Update stored posteriors
        self.posterior_ratios = new_posteriors.copy()
        
        # 3. Update q messages
        # q_{vc} = new_posterior - r_{cv}
        # r_csc.data contains r_{cv} values. 
        # r_csc.indices contains row indices (checks).
        
        indptr = r_csc.indptr
        indices = r_csc.indices
        data = r_csc.data
        
        q_data_csc = np.zeros_like(data)
        
        for j in range(self.num_vars):
            start = indptr[j]
            end = indptr[j+1]
            if start == end:
                continue
                
            total = new_posteriors[j]
            # q_{vc} = total - r_{cv}
            q_data_csc[start:end] = total - data[start:end]
            
        # Create new CSC matrix for Q
        q_csc = csc_matrix((q_data_csc, indices, indptr), shape=self.check_matrix.shape)
        
        # Convert back to CSR to store
        self.q_msgs = q_csc.tocsr()

    def step(self, syndrome):
        self.compute_check_to_variable(syndrome)
        self.compute_variable_to_check()
        self.current_iteration += 1
        
    def get_hard_decision(self):
        # Negative LLR means likelihood of error is higher (in some conventions).
        # Standard convention: LLR = log(P(no error) / P(error))
        # So LLR < 0 implies Error.
        return (self.posterior_ratios <= 0).astype(np.uint8)

    def decode(self, syndrome):
        # Only initialize if it's the first run or if explicitly reset externally
        # But for standard use, we assume clean slate unless part of relay
        if self.current_iteration == 0:
            self.initialize_decoder()
        
        for _ in range(self.max_iter):
            self.step(syndrome)
            
            # Check convergence
            candidate = self.get_hard_decision()
            # H * candidate = syndrome ?
            calc_syndrome = (self.check_matrix @ candidate) % 2
            if np.array_equal(calc_syndrome, syndrome):
                self.decoding = candidate
                return True, candidate
                
        self.decoding = self.get_hard_decision()
        return False, self.decoding

class RelayDecoder:
    def __init__(self, check_matrix, error_priors, 
                 pre_iter=80, num_legs=300, iter_per_leg=60,
                 gamma_interval=(-0.24, 0.66), gamma0=0.1):
        """
        Relay-BP Decoder.
        
        Args:
            check_matrix: Parity check matrix (scipy sparse or dense).
            error_priors: Error probabilities.
            pre_iter: Iterations for the first leg.
            num_legs: Maximum number of relay legs.
            iter_per_leg: Iterations per subsequent leg.
            gamma_interval: (min, max) for random memory strength.
            gamma0: Initial memory strength for the first leg.
        """
        self.bp_decoder = MinSumBPDecoder(
            check_matrix, error_priors, 
            max_iter=pre_iter, 
            gamma0=gamma0
        )
        self.pre_iter = pre_iter
        self.num_legs = num_legs
        self.iter_per_leg = iter_per_leg
        self.gamma_interval = gamma_interval
        self.gamma0 = gamma0
        
        self.num_vars = self.bp_decoder.num_vars
        
    def _generate_gammas(self):
        """Generate random memory strengths from uniform distribution."""
        low, high = self.gamma_interval
        return np.random.uniform(low, high, size=self.num_vars)
        
    def decode(self, syndrome):
        """
        Run the Relay-BP decoding process.
        Returns:
            success (bool): Whether a valid correction was found.
            correction (np.ndarray): The best correction found (lowest weight).
        """
        # Track best solution
        best_correction = None
        min_weight = float('inf')
        
        # 1. Run First Leg (Leg 0)
        # Using gamma0 (usually small positive, e.g. 0.1)
        self.bp_decoder.max_iter = self.pre_iter
        self.bp_decoder.set_memory_strengths(np.full(self.num_vars, self.gamma0))
        self.bp_decoder.initialize_decoder()
        
        # We manually step to allow checking (though decode() does this too)
        # But we want to preserve state for next leg.
        success, correction = self.bp_decoder.decode(syndrome)
        
        if success:
            weight = np.sum(correction) # Hamming weight (simplification, ideally weighted by priors)
            # Paper uses weighted weight: sum(e_j * log((1-p)/p))
            # But simple Hamming is often sufficient for CSS unless p varies wildly.
            # Let's use the actual log prior weight.
            weight = np.sum(correction * self.bp_decoder.log_prior_ratios)
            
            if weight < min_weight:
                min_weight = weight
                best_correction = correction.copy()
                
            # If standard BP converges, we might just return?
            # Paper says: "The algorithm stops when R legs have executed or S solutions have been found"
            # If S=1, we stop here. Let's assume S=1 for now.
            return True, best_correction

        # 2. Relay Loop
        for leg in range(self.num_legs):
            # Update configuration for relay legs
            self.bp_decoder.max_iter = self.iter_per_leg
            
            # Generate new disordered memory strengths
            new_gammas = self._generate_gammas()
            self.bp_decoder.set_memory_strengths(new_gammas)
            
            # Reset messages but keep posteriors
            self.bp_decoder.reset_for_new_leg()
            
            # Run this leg
            success, correction = self.bp_decoder.decode(syndrome)
            
            if success:
                weight = np.sum(correction * self.bp_decoder.log_prior_ratios)
                if weight < min_weight:
                    min_weight = weight
                    best_correction = correction.copy()
                
                # We found a solution. 
                # If we want just one valid solution, we can return.
                # If we want the *best* solution, we might continue?
                # Usually stopping on first success is standard for speed.
                return True, best_correction
                
        # If we exhausted all legs
        if best_correction is not None:
            return True, best_correction
            
        return False, self.bp_decoder.decoding
