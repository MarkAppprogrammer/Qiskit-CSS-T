"""
Circuit generation for tile code quantum error correction simulations.

This module provides functions for:
- Creating qubit registries
- Generating noiseless Stim circuits
- Generating noisy circuits with various noise models
"""

import stim
import numpy as np


def create_qubit_registry(Hx_positions, Hz_positions, Lx, Ly, grafted_qubits):
    """
    Create a registry mapping qubit indices to their types and coordinates.
    
    Args:
        Hx_positions: List of (i, j) positions for X stabilizers
        Hz_positions: List of (i, j) positions for Z stabilizers
        Lx, Ly: Lattice dimensions
        grafted_qubits: List of grafted qubit indices
        
    Returns:
        dict: Registry mapping qubit index to (type, index, (x, y))
    """
    registry = {}

    #### 1. Data qubits ####
    for n in range(2*Lx*Ly):
        if n in grafted_qubits:
            continue
        b = n // Lx
        a = n % Lx

        if b < Ly:
            registry[n] = ("DataL", n, (0.5+a, b))
        else:
            b = b - Ly
            registry[n] = ("DataR", n, (a, 0.5+b))

    #### 2. X checks ####
    for n, (i, j) in enumerate(Hx_positions):
        registry[n + 2*Lx*Ly] = ("XCheck", n, (i - Lx + 0.5, j))

    #### 3. Z checks ####
    for n, (i, j) in enumerate(Hz_positions):
        registry[n + 2*Lx*Ly + len(Hx_positions)] = ("ZCheck", n, (i, j - Ly + 0.5))
        
    return registry


def generate_tilecode_noiseless(
        rounds,
        qcode_res # dictionary containing information about the code
        ):
    """
    Generate a noiseless tilecode circuit without any error models.
    This circuit contains only the ideal quantum operations.

    Args:
        rounds: Number of syndrome measurement rounds
        qcode_res: Dictionary containing code information

    Returns:
        stim.Circuit: Noiseless circuit
    """
    # Extract information from qcode_res
    Hx_full_op = qcode_res['Hx_full_op']
    Hz_full_op = qcode_res['Hz_full_op']
    logical_X = qcode_res['logical_X']
    logical_Z = qcode_res['logical_Z']
    logical_X_grafted = qcode_res['logical_X_grafted']
    logical_Z_grafted = qcode_res['logical_Z_grafted']
    Hx_positions = qcode_res['Hx_positions']
    Hz_positions = qcode_res['Hz_positions']
    stab_size_x = qcode_res['stab_size_x']
    stab_size_y = qcode_res['stab_size_y']
    num_qubits = qcode_res['num_qubits']
    Lx = qcode_res['Lx']
    Ly = qcode_res['Ly']
    grafted_qubits = qcode_res['grafted_qubits']
    n_data = num_qubits

    circuit = stim.Circuit()

    ########### Generate circuit ###########
    ### 0. Generate coordinates ###
    qubit_registry = create_qubit_registry(Hx_positions, Hz_positions, Lx, Ly, grafted_qubits)
    
    DataL_keys = [q for q, (name, idx, (x, y)) in qubit_registry.items() if name == "DataL"]
    DataR_keys = [q for q, (name, idx, (x, y)) in qubit_registry.items() if name == "DataR"]
    XCheck_keys = [q for q, (name, idx, (x, y)) in qubit_registry.items() if name == "XCheck"]
    ZCheck_keys = [q for q, (name, idx, (x, y)) in qubit_registry.items() if name == "ZCheck"]
    
    def get_data_qubits_by_value_and_type(full_op_matrix, check_keys, qubit_registry, value, data_type):
        result_keys = []
        
        DEBUGING = 0
        for check_key in check_keys:
            idx = qubit_registry[check_key][1]
            if DEBUGING != idx: print("Something is not right!")
            DEBUGING += 1

            data_indices = np.where(full_op_matrix[idx] == value)[0]
            
            if data_type == "DataL":
                data_indices = data_indices[data_indices < Lx*Ly]
            elif data_type == "DataR":
                data_indices = data_indices[data_indices >= Lx*Ly]
            else:
                raise Exception(f"Unknown data type {data_type}.")


            if len(data_indices) > 1: 
                raise Exception(f"Multiple data qubits are connected to a single check. Something is wrong.")
            if len(data_indices) == 0: 
                continue

            if data_indices in grafted_qubits: 
                continue

            data_key = data_indices[0]

            if qubit_registry[data_key][0] != data_type:
                raise Exception(f"Data qubit is not {data_type}. Got {qubit_registry[data_key][0]} for data_key = {data_key}. Something is wrong.")
            
            result_keys.append((data_key, check_key))
        
        return result_keys

    # Usage
    R_Z1_keys = get_data_qubits_by_value_and_type(Hz_full_op, ZCheck_keys, qubit_registry, 1, "DataR")
    R_Z2_keys = get_data_qubits_by_value_and_type(Hz_full_op, ZCheck_keys, qubit_registry, 2, "DataR")
    R_Z3_keys = get_data_qubits_by_value_and_type(Hz_full_op, ZCheck_keys, qubit_registry, 3, "DataR")

    R_X1_keys = get_data_qubits_by_value_and_type(Hx_full_op, XCheck_keys, qubit_registry, 1, "DataR")
    R_X2_keys = get_data_qubits_by_value_and_type(Hx_full_op, XCheck_keys, qubit_registry, 2, "DataR")
    R_X3_keys = get_data_qubits_by_value_and_type(Hx_full_op, XCheck_keys, qubit_registry, 3, "DataR")

    L_Z1_keys = get_data_qubits_by_value_and_type(Hz_full_op, ZCheck_keys, qubit_registry, 1, "DataL")
    L_Z2_keys = get_data_qubits_by_value_and_type(Hz_full_op, ZCheck_keys, qubit_registry, 2, "DataL")
    L_Z3_keys = get_data_qubits_by_value_and_type(Hz_full_op, ZCheck_keys, qubit_registry, 3, "DataL")

    L_X1_keys = get_data_qubits_by_value_and_type(Hx_full_op, XCheck_keys, qubit_registry, 1, "DataL")
    L_X2_keys = get_data_qubits_by_value_and_type(Hx_full_op, XCheck_keys, qubit_registry, 2, "DataL")
    L_X3_keys = get_data_qubits_by_value_and_type(Hx_full_op, XCheck_keys, qubit_registry, 3, "DataL")
    

    for q, (name, idx, (x, y)) in qubit_registry.items():
        circuit.append("QUBIT_COORDS", [q], [x, y])

    ### 1. Reset ###
    circuit.append("R", DataL_keys + DataR_keys)
    circuit.append("TICK")

    ### 2. Repeat rounds ###
    for r in range(rounds):
        # 2-a. Before round (no depolarization in noiseless circuit)
        circuit.append("TICK")

        # 2-b. Stabilizer measurement cycle
        ### 2-b-1. TICK1
        circuit.append("RX", XCheck_keys)
        for d, a in R_Z1_keys: circuit.append("CNOT", [d, a])
        circuit.append("TICK")

        ### 2-b-2. TICK2
        for d, a in L_X2_keys: circuit.append("CNOT", [a, d])
        for d, a in R_Z3_keys: circuit.append("CNOT", [d, a])
        circuit.append("TICK")

        ### 2-b-3. TICK3
        for d, a in R_X2_keys: circuit.append("CNOT", [a, d])
        for d, a in L_Z1_keys: circuit.append("CNOT", [d, a])
        circuit.append("TICK")

        ### 2-b-4. TICK4
        for d, a in R_X1_keys: circuit.append("CNOT", [a, d])
        for d, a in L_Z2_keys: circuit.append("CNOT", [d, a])
        circuit.append("TICK")

        ### 2-b-5. TICK5
        for d, a in R_X3_keys: circuit.append("CNOT", [a, d])
        for d, a in L_Z3_keys: circuit.append("CNOT", [d, a])
        circuit.append("TICK")

        ### 2-b-6. TICK6
        for d, a in L_X1_keys: circuit.append("CNOT", [a, d])
        for d, a in R_Z2_keys: circuit.append("CNOT", [d, a])
        circuit.append("TICK")

        ### 2-b-7. TICK7
        for d, a in L_X3_keys: circuit.append("CNOT", [a, d])
        circuit.append("MZ", ZCheck_keys)
        circuit.append("I", DataR_keys)
        circuit.append("TICK")

        ### 2-b-8. TICK8
        circuit.append("MX", XCheck_keys)
        circuit.append("RZ", ZCheck_keys)
        circuit.append("I", DataL_keys)
        circuit.append("I", DataR_keys)
        circuit.append("TICK")

        # 2-c. DETECTOR
        # for i in range(len(ZCheck_keys + XCheck_keys)):
        for i in range(len(XCheck_keys), len(ZCheck_keys + XCheck_keys)):
            if r == 0:
                circuit.append("DETECTOR", [stim.target_rec(-i-1)])
            else:
                circuit.append("DETECTOR", [
                    stim.target_rec(-i-1),
                    stim.target_rec(-i-1 - len(ZCheck_keys + XCheck_keys))
                ])

    # 3. Include observable
    circuit.append("M", DataL_keys + DataR_keys)
    for ZCheck_idx, ZCheck_key in enumerate(ZCheck_keys):
        # 해당 ZCheck가 측정하는 data qubit을 찾자.
        support_data_qubits_idx = []
        for idx, (q, (name, _, (x, y))) in enumerate(qubit_registry.items()):
            if name == "DataL" or name == "DataR":
                if Hz_full_op[ZCheck_idx, q] != 0:
                    support_data_qubits_idx.append(idx)
        circuit.append("DETECTOR",
            [stim.target_rec( -len(ZCheck_keys + XCheck_keys+ DataL_keys + DataR_keys) + ZCheck_idx)] +
            [stim.target_rec( -len(DataL_keys + DataR_keys) + support_data_qubit_idx) for support_data_qubit_idx in support_data_qubits_idx]
        )
    # for XCheck_idx, XCheck_key in enumerate(XCheck_keys):
    #     support_data_qubits_idx = []
    #     for idx, (q, (name, _, (x, y))) in enumerate(qubit_registry.items()):
    #         if name == "DataL" or name == "DataR":
    #             if Hx_full_op[XCheck_idx, q] != 0:
    #                 support_data_qubits_idx.append(idx)
    #     circuit.append("DETECTOR",
    #         [stim.target_rec( -len(XCheck_keys+ DataL_keys + DataR_keys) + XCheck_idx)] +
    #         [stim.target_rec( -len(DataL_keys + DataR_keys) + support_data_qubit_idx) for support_data_qubit_idx in support_data_qubits_idx]
    #     )

    logical_Z_grafted = logical_Z_grafted.todense().tolist()
    logical_X_grafted = logical_X_grafted.todense().tolist()
    k = len(logical_Z_grafted)

    for kn, lz in enumerate(logical_Z_grafted):
        logical_Z_indices = [i for i, val in enumerate(lz) if val == 1]
        circuit.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(-len(DataL_keys + DataR_keys) + i) for i in logical_Z_indices],
            kn
        )
    # for kn, lx in enumerate(logical_X_grafted):
    #     logical_X_indices = [i for i, val in enumerate(lx) if val == 1]
    #     circuit.append(
    #         "OBSERVABLE_INCLUDE",
    #         [stim.target_rec(-len(DataL_keys + DataR_keys) + i) for i in logical_X_indices],
    #         k + kn
    #     )

    return circuit


def generate_tilecode(
        rounds,
        qcode_res,
        noise_model='error_rate',
        # Standard noise model parameters
        after_clifford_depolarization=0.001,
        after_reset_flip_probability=0.002,
        before_measure_flip_probability=0.003,
        before_round_data_depolarization=0.004,
        # SI1000 noise model parameter
        probability=0.001,
        # Bravyi noise model parameter
        error_rate=0.001
):
    """
    Generate a tilecode circuit with the specified noise model.

    Args:
        rounds: Number of syndrome measurement rounds
        qcode_res: Dictionary containing code information
        noise_model: 'standard', 'si1000', or 'bravyi' (default: 'standard')

        For 'standard' noise model:
            after_clifford_depolarization: Error probability after CNOT gates
            after_reset_flip_probability: Error probability after reset operations
            before_measure_flip_probability: Error probability before measurements
            before_round_data_depolarization: Error probability before each round

        For 'si1000' noise model:
            probability: Base error probability (p)

        For 'bravyi' noise model:
            error_rate: Base error rate (all operations use this with different coefficients)

    Returns:
        stim.Circuit: Noisy circuit with the specified noise model
    """
    from .noise_models import (
        standard_depolarizing_noise_model,
        si1000_noise_model,
        bravyi_noise_model
    )

    # Generate noiseless circuit
    noiseless_circuit = generate_tilecode_noiseless(rounds, qcode_res)

    # Apply noise model
    if noise_model == 'standard':
        return standard_depolarizing_noise_model(
            noiseless_circuit,
            qcode_res,
            after_clifford_depolarization,
            after_reset_flip_probability,
            before_measure_flip_probability,
            before_round_data_depolarization
        )
    elif noise_model == 'si1000':
        return si1000_noise_model(
            noiseless_circuit,
            qcode_res,
            probability
        )
    elif noise_model == 'bravyi':
        return bravyi_noise_model(
            noiseless_circuit,
            qcode_res,
            error_rate
        )
    else:
        raise ValueError(f"Unknown noise model: {noise_model}. Choose 'standard', 'si1000', or 'bravyi'.")