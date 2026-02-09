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
    """
    Helper to append operations to a list, NOT directly to circuit.
    """
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
                                 schedule: Optional[dict] = None) -> stim.Circuit:
    is_self_dual = (Hx.shape == Hz.shape) and np.array_equal(Hx, Hz)
    
    if is_self_dual:
        return _generate_self_dual_schedule(Hx, rounds, memory_basis, schedule)
    else:
        return _generate_standard_schedule(Hx, Hz, rounds, memory_basis, schedule)

def _generate_self_dual_schedule(H, rounds, memory_basis, schedule):
    # (Kept unchanged as it was working for self-dual codes)
    num_data = H.shape[1]
    num_checks = H.shape[0]
    data_qubits = list(range(num_data))
    ancillas = list(range(num_data, num_data + num_checks))
    circuit = stim.Circuit()
    
    for i, q in enumerate(data_qubits): circuit.append("QUBIT_COORDS", [q], [i * 2.0, 2.0])
    anc_start_x = get_start_x(num_checks, 4.0, ((num_data - 1) * 2.0) / 2.0)
    anc_coords = []
    for i, q in enumerate(ancillas):
        pos = [anc_start_x + i * 4.0, 4.0]
        circuit.append("QUBIT_COORDS", [q], pos); anc_coords.append(pos)

    circuit.append("R" if memory_basis == "Z" else "RX", data_qubits)
    circuit.append("R", ancillas); circuit.append("TICK")

    def append_round(c, is_first):
        c.append("H", ancillas); c.append("TICK")
        
        # X-Checks
        x_ops = append_scheduled_cnots(c, H, ancillas, data_qubits, False, schedule)
        x_ops.sort(key=lambda x: x[0])
        for p, ctrl, targ in x_ops: c.append("CNOT", [ctrl, targ])
            
        c.append("H", ancillas); c.append("M", ancillas); c.append("R", ancillas); c.append("TICK")
        
        # Z-Checks
        z_ops = append_scheduled_cnots(c, H, ancillas, data_qubits, True, schedule)
        z_ops.sort(key=lambda x: x[0])
        for p, ctrl, targ in z_ops: c.append("CNOT", [ctrl, targ])
            
        c.append("M", ancillas); c.append("R", ancillas); c.append("TICK")
        
        total_m = 2 * num_checks
        for i in range(num_checks): 
            rec_now = stim.target_rec(-total_m + i); rec_prev = stim.target_rec(-total_m * 2 + i)
            args = [rec_now, rec_prev] if not is_first else [rec_now]
            if not is_first or memory_basis == "X": c.append("DETECTOR", args, [anc_coords[i][0], anc_coords[i][1], 0])
        for i in range(num_checks): 
            rec_now = stim.target_rec(-num_checks + i); rec_prev = stim.target_rec(-num_checks - total_m + i)
            args = [rec_now, rec_prev] if not is_first else [rec_now]
            if not is_first or memory_basis == "Z": c.append("DETECTOR", args, [anc_coords[i][0], anc_coords[i][1], 1])

    append_round(circuit, True)
    if rounds > 1:
        loop = stim.Circuit(); append_round(loop, False); circuit.append(stim.CircuitRepeatBlock(rounds - 1, loop))

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
    # UPDATED: Split measurement schedule to be more robust for 3D codes
    # This serializes X and Z checks to prevent non-deterministic detector issues
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
        
        # X-checks: Ancilla is Control
        x_ops = append_scheduled_cnots(c, Hx, x_ancillas, data_qubits, True, schedule, schedule_offset=0)
        x_ops.sort(key=lambda x: x[0])
        for p, ctrl, targ in x_ops:
            c.append("CNOT", [ctrl, targ])
        c.append("TICK")
        
        # Measure X (Immediate measurement to clear entanglement)
        c.append("H", x_ancillas)
        c.append("TICK")
        c.append("MR", x_ancillas) # X ancillas measured
        
        # --- BLOCK 2: Z Stabilizers ---
        # Z-checks: Ancilla is Target (offset schedule index by num_x)
        z_ops = append_scheduled_cnots(c, Hz, z_ancillas, data_qubits, False, schedule, schedule_offset=num_x)
        z_ops.sort(key=lambda x: x[0])
        for p, ctrl, targ in z_ops:
            c.append("CNOT", [ctrl, targ])
        c.append("TICK")
        
        # Measure Z
        c.append("MR", z_ancillas)
        
        # --- Detectors ---
        # Note: X measurements happened FIRST (indices -num_z -num_x ... -num_z)
        # Z measurements happened SECOND (indices -num_z ... -1)
        
        tot_meas = num_x + num_z
        
        # X Detectors
        for i in range(num_x):
            # Rec index for current round X measurement
            rec_now = stim.target_rec(-tot_meas + i) # X measures are at start of block
            rec_prev = stim.target_rec(-tot_meas * 2 + i)
            
            args = [rec_now, rec_prev] if not is_first else [rec_now]
            if not is_first or memory_basis == "X": 
                c.append("DETECTOR", args, [x_coords[i][0], x_coords[i][1], 0])
        
        # Z Detectors
        for i in range(num_z):
            # Rec index for current round Z measurement
            rec_now = stim.target_rec(-num_z + i) # Z measures are at end of block
            rec_prev = stim.target_rec(-tot_meas -num_z + i)
            
            args = [rec_now, rec_prev] if not is_first else [rec_now]
            if not is_first or memory_basis == "Z": 
                c.append("DETECTOR", args, [z_coords[i][0], z_coords[i][1], 0])

    append_round(circuit, True)
    if rounds > 1: 
        loop = stim.Circuit(); append_round(loop, False); circuit.append(stim.CircuitRepeatBlock(rounds - 1, loop))
        
    # Final Measurement of Data Qubits
    if memory_basis == "Z":
        circuit.append("M", data_qubits)
        # Check Z stabilizers against final data measurements
        for i, row in enumerate(Hz):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            # Compare to last Z ancilla measurement
            rec.append(stim.target_rec(-(num_data + num_z - i))) 
            circuit.append("DETECTOR", rec, [z_coords[i][0], z_coords[i][1], 0])
            
    elif memory_basis == "X":
        circuit.append("MX", data_qubits)
        # Check X stabilizers
        for i, row in enumerate(Hx):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            # Compare to last X ancilla measurement
            # X ancillas were measured num_z steps before the Z ancillas
            rec.append(stim.target_rec(-(num_data + num_z + num_x - i)))
            circuit.append("DETECTOR", rec, [x_coords[i][0], x_coords[i][1], 0])

    op = find_logical_operator(Hx, Hz, basis=memory_basis)
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-(num_data - k)) for k in np.flatnonzero(op)], 0)
    return circuit

def generate_experiment_with_noise(Hx, Hz, rounds, noise_model_name, noise_params, memory_basis="Z", schedule=None):
    from noise_models import standard_depolarizing_noise_model, si1000_noise_model
    
    clean_circuit = generate_css_memory_experiment(Hx, Hz, rounds, memory_basis=memory_basis, schedule=schedule)
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