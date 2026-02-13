import stim
import numpy as np
import json
from typing import List, Tuple, Dict, Optional
from helpers import find_logical_operator

# --- Scheduling Helpers ---

def load_schedule(filepath: str) -> Dict[Tuple[int, int], int]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    schedule = {}
    for k, v in data.items():
        parts = k.split(',')
        schedule[(int(parts[0]), int(parts[1]))] = v
    return schedule

def get_start_x(num_items: int, spacing: float, center: float) -> float:
    return center - ((num_items - 1) * spacing / 2.0) if num_items > 1 else center

def append_scheduled_cnots(c: stim.Circuit, 
                           check_matrix: np.ndarray, 
                           ancilla_idxs: List[int], 
                           data_idxs: List[int], 
                           ancilla_is_control: bool, 
                           schedule: Optional[Dict[Tuple[int, int], int]],
                           schedule_offset: int = 0):

    ops = [] 
    for i, row in enumerate(check_matrix):
        targets = [data_idxs[q] for q in np.flatnonzero(row)]
        anc = ancilla_idxs[i]
        for t_idx, dat in enumerate(targets):
            key = (i + schedule_offset, data_idxs.index(dat))
            priority = schedule.get(key, t_idx) if schedule else t_idx
            
            if ancilla_is_control: 
                ops.append((priority, anc, dat))
            else: 
                ops.append((priority, dat, anc))
    return ops

# --- Circuit Generators ---

def generate_css_memory_experiment(Hx: np.ndarray, 
                                 Hz: np.ndarray, 
                                 rounds: int, 
                                 memory_basis: str = "Z", 
                                 schedule: Optional[dict] = None,
                                 code_type: str = "unknown") -> stim.Circuit:
    
    # Check strict self-duality (matrices must be identical)
    is_strictly_self_dual = (Hx.shape == Hz.shape) and np.array_equal(Hx, Hz)
    
    # 1. Color Code Schedule (C_XYZ)
    # Applied to 'triangular' and 'square' codes (which are geometric color codes)
    if code_type in ["triangular", "square"]:
        if not is_strictly_self_dual:
             # Fallback or warning: Color codes must be self-dual for C_XYZ
             print(f"Warning: {code_type} requested but matrices are not identical. Attempting standard schedule.")
             return _generate_standard_schedule(Hx, Hz, rounds, memory_basis, schedule)
        return generate_color_code_experiment(Hx, rounds, schedule)

    # 2. Standard Self-Dual Schedule (X-check then Z-check)
    # Applied to 'self_dual', 'dual_containing', or any other self-dual code not specified above
    elif is_strictly_self_dual:
        return _generate_self_dual_schedule(Hx, rounds, memory_basis, schedule)

    # 3. Standard Non-Dual Schedule (Separate X and Z matrices)
    else:
        return _generate_standard_schedule(Hx, Hz, rounds, memory_basis, schedule)

def generate_color_code_experiment(H: np.ndarray, 
                                   rounds: int, 
                                   schedule: Optional[dict] = None) -> stim.Circuit:
    """
    Generates a Color Code memory experiment using the C_XYZ cycle.
    Used for 'triangular' and 'square' codes.
    
    Cycle:
    1. Measure Z-stabilizers (Ancilla=Target).
    2. Apply C_XYZ to Data Qubits (Z->X, X->Y, Y->Z).
    3. Detectors compare Round[t] with Round[t-3].
    """
    num_data = H.shape[1]
    num_checks = H.shape[0]
    data_qubits = list(range(num_data))
    ancillas = list(range(num_data, num_data + num_checks))
    
    circuit = stim.Circuit()
    
    # Coordinates
    for i, q in enumerate(data_qubits):
        circuit.append("QUBIT_COORDS", [q], [i * 2.0, 2.0])
    
    anc_start_x = get_start_x(num_checks, 4.0, ((num_data - 1) * 2.0) / 2.0)
    anc_coords = []
    for i, q in enumerate(ancillas):
        pos = [anc_start_x + i * 4.0, 4.0]
        circuit.append("QUBIT_COORDS", [q], pos)
        anc_coords.append(pos)

    # Initialization
    circuit.append("R", data_qubits) # Start in Z-basis
    circuit.append("R", ancillas)
    circuit.append("TICK")

    # Round Logic
    def append_round(c, round_idx):
        # A. Measure Stabilizers (Always measure "Z" of current frame)
        ops = append_scheduled_cnots(c, H, ancillas, data_qubits, 
                                     ancilla_is_control=False,
                                     schedule=schedule, 
                                     schedule_offset=0)
        ops.sort(key=lambda x: x[0])
        for p, ctrl, targ in ops:
            c.append("CNOT", [ctrl, targ])
            
        c.append("MR", ancillas)
        
        # B. Cycle Basis (C_XYZ)
        c.append("C_XYZ", data_qubits)
        c.append("TICK")
        
        # C. Detectors (Period 3)
        if round_idx >= 3:
            for i in range(num_checks):
                rec_now = stim.target_rec(-num_checks + i)
                rec_prev = stim.target_rec(-num_checks * 4 + i)
                basis_label = round_idx % 3
                c.append("DETECTOR", [rec_now, rec_prev], 
                         [anc_coords[i][0], anc_coords[i][1], basis_label])

    # Loop Construction
    for r in range(rounds):
        if r < 3:
            append_round(circuit, r)
        else:
            # Unrolled for clarity with round_idx logic
            append_round(circuit, r)

    # Final Measurement
    final_basis_idx = rounds % 3 # 0=Z, 1=X, 2=Y
    
    if final_basis_idx == 0:   # Physical Z
        circuit.append("M", data_qubits)
    elif final_basis_idx == 1: # Physical X
        circuit.append("MX", data_qubits)
    elif final_basis_idx == 2: # Physical Y
        circuit.append("MY", data_qubits)
    
    # Final Detectors
    if rounds >= 3:
        for i, row in enumerate(H):
            rec_targets = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            # Last matching check was 3 rounds ago (offset by data meas + 3 rounds of ancillas)
            rec_prev = stim.target_rec(-(num_data + num_checks * 3 - i))
            rec_targets.append(rec_prev)
            circuit.append("DETECTOR", rec_targets, 
                           [anc_coords[i][0], anc_coords[i][1], final_basis_idx])

    # Observable
    op = find_logical_operator(H, H, basis="Z")
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-(num_data - k)) for k in np.flatnonzero(op)], 0)
    
    return circuit

def _generate_self_dual_schedule(H, rounds, memory_basis, schedule):

    num_data = H.shape[1]
    num_checks = H.shape[0]
    data_qubits = list(range(num_data))
    ancillas = list(range(num_data, num_data + num_checks))
    circuit = stim.Circuit()
    
    # Coordinates
    for i, q in enumerate(data_qubits): 
        circuit.append("QUBIT_COORDS", [q], [i * 2.0, 2.0])
    
    anc_start_x = get_start_x(num_checks, 4.0, ((num_data - 1) * 2.0) / 2.0)
    anc_coords = []
    for i, q in enumerate(ancillas):
        pos = [anc_start_x + i * 4.0, 4.0]
        circuit.append("QUBIT_COORDS", [q], pos)
        anc_coords.append(pos)

    # Initialization
    circuit.append("R" if memory_basis == "Z" else "RX", data_qubits)
    circuit.append("R", ancillas)
    circuit.append("TICK")

    def append_round(c, is_first):
        # --- X-Checks ---
        c.append("H", ancillas)
        c.append("TICK")
        
        # X-checks require Ancilla to be CONTROL
        x_ops = append_scheduled_cnots(c, H, ancillas, data_qubits, 
                                     ancilla_is_control=True, 
                                     schedule=schedule, 
                                     schedule_offset=0)
        x_ops.sort(key=lambda x: x[0])
        for p, ctrl, targ in x_ops: 
            c.append("CNOT", [ctrl, targ])
            
        c.append("H", ancillas)
        c.append("M", ancillas)
        c.append("R", ancillas)
        c.append("TICK")
        
        # --- Z-Checks ---
        # Z-checks require Ancilla to be TARGET
        z_ops = append_scheduled_cnots(c, H, ancillas, data_qubits, 
                                     ancilla_is_control=False, 
                                     schedule=schedule, 
                                     schedule_offset=num_checks)
        z_ops.sort(key=lambda x: x[0])
        for p, ctrl, targ in z_ops: 
            c.append("CNOT", [ctrl, targ])
            
        c.append("M", ancillas)
        c.append("R", ancillas)
        c.append("TICK")
        
        # --- Detectors ---
        total_m = 2 * num_checks 
        
        # X-Detectors (Measurements happened 'num_checks' ago)
        for i in range(num_checks): 
            rec_now = stim.target_rec(-total_m + i)
            rec_prev = stim.target_rec(-total_m * 2 + i)
            args = [rec_now, rec_prev] if not is_first else [rec_now]
            if not is_first or memory_basis == "X": 
                c.append("DETECTOR", args, [anc_coords[i][0], anc_coords[i][1], 0])
        
        # Z-Detectors (Measurements just happened)
        for i in range(num_checks): 
            rec_now = stim.target_rec(-num_checks + i)
            rec_prev = stim.target_rec(-num_checks - total_m + i)
            args = [rec_now, rec_prev] if not is_first else [rec_now]
            if not is_first or memory_basis == "Z": 
                c.append("DETECTOR", args, [anc_coords[i][0], anc_coords[i][1], 1])

    append_round(circuit, True)
    if rounds > 1:
        loop = stim.Circuit()
        append_round(loop, False)
        circuit.append(stim.CircuitRepeatBlock(rounds - 1, loop))

    # Final Measurement
    if memory_basis == "Z":
        circuit.append("M", data_qubits)
        for i, row in enumerate(H):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            rec.append(stim.target_rec(-(num_data + num_checks - i))) 
            circuit.append("DETECTOR", rec, [anc_coords[i][0], anc_coords[i][1], 1])
    else:
        circuit.append("MX", data_qubits)
        for i, row in enumerate(H):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            rec.append(stim.target_rec(-(num_data + 2*num_checks - i)))
            circuit.append("DETECTOR", rec, [anc_coords[i][0], anc_coords[i][1], 0])

    op = find_logical_operator(H, H, basis=memory_basis)
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-(num_data - k)) for k in np.flatnonzero(op)], 0)
    return circuit

def _generate_standard_schedule(Hx, Hz, rounds, memory_basis, schedule):
    """
    Generates standard schedule for non-self-dual codes (separate Hx and Hz).
    """
    num_data = Hx.shape[1]
    num_x = Hx.shape[0]
    num_z = Hz.shape[0]
    data_qubits = list(range(num_data))
    x_ancillas = list(range(num_data, num_data + num_x))
    z_ancillas = list(range(num_data + num_x, num_data + num_x + num_z))
    circuit = stim.Circuit()
    
    # Coords
    x_start = get_start_x(num_x, 4.0, ((num_data - 1) * 2.0)/2.0)
    x_coords = []
    for i, q in enumerate(x_ancillas): 
        pos = [x_start + i * 4.0, 0]; circuit.append("QUBIT_COORDS", [q], pos); x_coords.append(pos)
    for i, q in enumerate(data_qubits): 
        circuit.append("QUBIT_COORDS", [q], [i * 2.0, 2.0])
    z_start = get_start_x(num_z, 4.0, ((num_data - 1) * 2.0)/2.0)
    z_coords = []
    for i, q in enumerate(z_ancillas): 
        pos = [z_start + i * 4.0, 4.0]; circuit.append("QUBIT_COORDS", [q], pos); z_coords.append(pos)

    # Init
    circuit.append("R" if memory_basis == "Z" else "RX", data_qubits)
    circuit.append("R", x_ancillas + z_ancillas)
    circuit.append("TICK")

    def append_round(c, is_first):
        # --- BLOCK 1: X Stabilizers ---
        c.append("H", x_ancillas)
        c.append("TICK")
        x_ops = append_scheduled_cnots(c, Hx, x_ancillas, data_qubits, True, schedule, schedule_offset=0)
        x_ops.sort(key=lambda x: x[0])
        for p, ctrl, targ in x_ops: c.append("CNOT", [ctrl, targ])
        c.append("TICK"); c.append("H", x_ancillas); c.append("TICK"); c.append("MR", x_ancillas)
        
        # --- BLOCK 2: Z Stabilizers ---
        z_ops = append_scheduled_cnots(c, Hz, z_ancillas, data_qubits, False, schedule, schedule_offset=num_x)
        z_ops.sort(key=lambda x: x[0])
        for p, ctrl, targ in z_ops: c.append("CNOT", [ctrl, targ])
        c.append("TICK"); c.append("MR", z_ancillas)
        
        # --- Detectors ---
        tot_meas = num_x + num_z
        for i in range(num_x):
            rec_now = stim.target_rec(-tot_meas + i)
            rec_prev = stim.target_rec(-tot_meas * 2 + i)
            args = [rec_now, rec_prev] if not is_first else [rec_now]
            if not is_first or memory_basis == "X": c.append("DETECTOR", args, [x_coords[i][0], x_coords[i][1], 0])
        for i in range(num_z):
            rec_now = stim.target_rec(-num_z + i)
            rec_prev = stim.target_rec(-tot_meas -num_z + i)
            args = [rec_now, rec_prev] if not is_first else [rec_now]
            if not is_first or memory_basis == "Z": c.append("DETECTOR", args, [z_coords[i][0], z_coords[i][1], 0])

    append_round(circuit, True)
    if rounds > 1: 
        loop = stim.Circuit(); append_round(loop, False); circuit.append(stim.CircuitRepeatBlock(rounds - 1, loop))
        
    # Final Measurement
    if memory_basis == "Z":
        circuit.append("M", data_qubits)
        for i, row in enumerate(Hz):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            rec.append(stim.target_rec(-(num_data + num_z - i))) 
            circuit.append("DETECTOR", rec, [z_coords[i][0], z_coords[i][1], 0])
    elif memory_basis == "X":
        circuit.append("MX", data_qubits)
        for i, row in enumerate(Hx):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            rec.append(stim.target_rec(-(num_data + num_z + num_x - i)))
            circuit.append("DETECTOR", rec, [x_coords[i][0], x_coords[i][1], 0])

    op = find_logical_operator(Hx, Hz, basis=memory_basis)
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-(num_data - k)) for k in np.flatnonzero(op)], 0)
    return circuit

def generate_experiment_with_noise(Hx, Hz, rounds, noise_model_name, noise_params, memory_basis="Z", schedule=None, code_type="unknown"):
    """
    Generates the full noisy experiment.
    Updated to accept 'code_type' and pass it to the circuit generator.
    """
    from noise_models import standard_depolarizing_noise_model, si1000_noise_model
    
    clean_circuit = generate_css_memory_experiment(
        Hx, Hz, rounds, 
        memory_basis=memory_basis, 
        schedule=schedule,
        code_type=code_type
    )
    
    num_data = Hx.shape[1]
    data_qubits = list(range(num_data))
    base_p = noise_params['p']
    
    if noise_model_name == "depolarizing":
        return standard_depolarizing_noise_model(
            circuit=clean_circuit, 
            data_qubits=data_qubits,
            after_clifford_depolarization=noise_params.get('p_clifford', base_p),
            after_reset_flip_probability=noise_params.get('p_reset', base_p),
            before_measure_flip_probability=noise_params.get('p_meas', base_p),
            before_round_data_depolarization=noise_params.get('p_data_round', base_p)
        )
    elif noise_model_name == "si1000":
        return si1000_noise_model(circuit=clean_circuit, data_qubits=data_qubits, probability=base_p)
    else: 
        raise ValueError(f"Unknown noise model: {noise_model_name}")