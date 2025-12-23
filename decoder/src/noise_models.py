"""
Noise models for tile code quantum error correction simulations.

This module provides various noise models that can be applied to Stim circuits:
- standard_depolarizing_noise_model: Standard depolarizing noise
- si1000_noise_model: SI1000 noise model inspired by superconducting transmon qubits
- bravyi_noise_model: Bravyi noise model following BivariateBicycleCodes approach
"""

import stim


def standard_depolarizing_noise_model(
        circuit: stim.Circuit,
        qcode_res: dict,
        after_clifford_depolarization: float,
        after_reset_flip_probability: float,
        before_measure_flip_probability: float,
        before_round_data_depolarization: float
) -> stim.Circuit:
    """
    Applies a standard depolarizing noise model to a noiseless Stim circuit.

    This noise model applies:
    - X_ERROR after initial R (reset) gates
    - DEPOLARIZE1 before each round on data qubits
    - DEPOLARIZE2 after CNOT gates
    - X_ERROR before and after MR (measure-reset) gates
    - X_ERROR before final M (measure) gates

    Args:
        circuit: The input noiseless quantum circuit
        qcode_res: Dictionary containing code information (to identify data qubits)
        after_clifford_depolarization: Error probability after CNOT gates
        after_reset_flip_probability: Error probability after reset operations
        before_measure_flip_probability: Error probability before measurements
        before_round_data_depolarization: Error probability before each round

    Returns:
        stim.Circuit: A new circuit with noise operations inserted
    """
    from .circuit import create_qubit_registry
    
    # Get data qubit information
    Hx_positions = qcode_res['Hx_positions']
    Hz_positions = qcode_res['Hz_positions']
    Lx = qcode_res['Lx']
    Ly = qcode_res['Ly']
    grafted_qubits = qcode_res['grafted_qubits']

    qubit_registry = create_qubit_registry(Hx_positions, Hz_positions, Lx, Ly, grafted_qubits)
    DataL_keys = [q for q, (name, _, _) in qubit_registry.items() if name == "DataL"]
    DataR_keys = [q for q, (name, _, _) in qubit_registry.items() if name == "DataR"]
    data_qubits = DataL_keys + DataR_keys

    result = stim.Circuit()

    # State tracking
    first_reset_seen = False
    tick_count = 0

    for instruction in circuit:
        # Handle repeat blocks recursively
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(
                repeat_count=instruction.repeat_count,
                body=standard_depolarizing_noise_model(
                    instruction.body_copy(),
                    qcode_res,
                    after_clifford_depolarization,
                    after_reset_flip_probability,
                    before_measure_flip_probability,
                    before_round_data_depolarization
                )))
        # Add X_ERROR after initial R (reset) gates
        elif instruction.name == 'R' and not first_reset_seen:
            result.append(instruction)
            result.append('X_ERROR', instruction.targets_copy(), after_reset_flip_probability)
            first_reset_seen = True
        # Add DEPOLARIZE2 after CNOT gates
        elif instruction.name == 'CNOT':
            result.append(instruction)
            result.append('DEPOLARIZE2', instruction.targets_copy(), after_clifford_depolarization)
        # Add X_ERROR before and after MR (measure-reset) gates
        elif instruction.name == 'MR':
            result.append('X_ERROR', instruction.targets_copy(), before_measure_flip_probability)
            result.append(instruction)
            result.append('X_ERROR', instruction.targets_copy(), after_reset_flip_probability)
        # Add X_ERROR before final M (measure) gates
        elif instruction.name == 'M':
            result.append('X_ERROR', instruction.targets_copy(), before_measure_flip_probability)
            result.append(instruction)
        # Handle TICK - add DEPOLARIZE1 at the start of each round
        elif instruction.name == 'TICK':
            result.append(instruction)
            tick_count += 1
            # After first TICK (tick_count==1), every 9 TICKs we're at the start of a new round
            # Pattern: TICK(1) -> [TICK(2) DEPOLARIZE1 TICK -> ... -> TICK(9)] -> repeat
            # So we add DEPOLARIZE1 after TICK 2, 11, 20, 29, ... (i.e., tick_count % 9 == 2)
            if first_reset_seen and tick_count >= 2 and (tick_count - 1) % 9 == 1:
                result.append('DEPOLARIZE1', data_qubits, before_round_data_depolarization)
                result.append('TICK')
                tick_count += 1
        else:
            result.append(instruction)

    return result


def si1000_noise_model(
        circuit: stim.Circuit,
        qcode_res: dict,
        probability: float
) -> stim.Circuit:
    """
    Applies the SI1000 noise model to a noiseless Stim circuit.

    This is a specialized noise model inspired by superconducting transmon qubit arrays
    with different error rates for different operations:
    - Reset operations: 2x base probability (X_ERROR)
    - Two-qubit gates (CNOT): p/10 base probability (DEPOLARIZE2)
    - Measurement operations: measurement flip 5x, depolarization p
    - Idle (before round): p/10 base probability (DEPOLARIZE1)

    Args:
        circuit: The input noiseless quantum circuit
        qcode_res: Dictionary containing code information (to identify data qubits)
        probability: Base error probability (p)

    Returns:
        stim.Circuit: A new circuit with noise operations inserted
    """
    from .circuit import create_qubit_registry
    
    # Get data qubit information
    Hx_positions = qcode_res['Hx_positions']
    Hz_positions = qcode_res['Hz_positions']
    Lx = qcode_res['Lx']
    Ly = qcode_res['Ly']
    grafted_qubits = qcode_res['grafted_qubits']

    qubit_registry = create_qubit_registry(Hx_positions, Hz_positions, Lx, Ly, grafted_qubits)
    DataL_keys = [q for q, (name, _, _) in qubit_registry.items() if name == "DataL"]
    DataR_keys = [q for q, (name, _, _) in qubit_registry.items() if name == "DataR"]
    XCheck_keys = [q for q, (name, _, _) in qubit_registry.items() if name == "XCheck"]
    ZCheck_keys = [q for q, (name, _, _) in qubit_registry.items() if name == "ZCheck"]
    data_qubits = DataL_keys + DataR_keys
    all_qubits = data_qubits + XCheck_keys + ZCheck_keys

    result = stim.Circuit()

    # State tracking
    first_reset_seen = False
    tick_count = 0

    for instruction in circuit:
        # Handle repeat blocks recursively
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(
                repeat_count=instruction.repeat_count,
                body=si1000_noise_model(
                    instruction.body_copy(),
                    qcode_res,
                    probability
                )))
        # Add X_ERROR after initial R (reset) gates with 2p
        elif instruction.name == 'R' and not first_reset_seen:
            result.append(instruction)
            result.append('X_ERROR', instruction.targets_copy(), 2 * probability)
            # Add DEPOLARIZE1 on idle qubits
            idle_qubits = list(set(all_qubits) - set(instruction.targets_copy()))
            if idle_qubits:
                result.append('DEPOLARIZE1', idle_qubits, 2 * probability)
            first_reset_seen = True
        # Add DEPOLARIZE2 after CNOT gates with p/10
        elif instruction.name == 'CNOT':
            result.append(instruction)
            result.append('DEPOLARIZE2', instruction.targets_copy(), probability / 10)
        # Add X_ERROR before MR with 5p (measurement flip) and after with 2p (reset)
        # Also add DEPOLARIZE1 with p on the measured qubits
        elif instruction.name == 'MR':
            # Before measurement: depolarization p
            result.append('DEPOLARIZE1', instruction.targets_copy(), probability)
            # Measurement with flip probability 5p
            # We need to split MR into M and R to add different errors
            # But Stim's MR is atomic, so we approximate:
            # Add X_ERROR before (acts as measurement flip)
            result.append('X_ERROR', instruction.targets_copy(), 5 * probability)
            result.append(instruction)
            # After reset: X_ERROR with 2p
            result.append('X_ERROR', instruction.targets_copy(), 2 * probability)
            # Idle qubits get DEPOLARIZE1 with 2p (ResonatorIdle)
            idle_qubits = list(set(all_qubits) - set(instruction.targets_copy()))
            if idle_qubits:
                result.append('DEPOLARIZE1', idle_qubits, 2 * probability)
        # Add X_ERROR before final M (measure) gates with 5p
        elif instruction.name == 'M':
            result.append('DEPOLARIZE1', instruction.targets_copy(), probability)
            result.append('X_ERROR', instruction.targets_copy(), 5 * probability)
            result.append(instruction)
        # Handle TICK - add DEPOLARIZE1 at the start of each round with p/10
        elif instruction.name == 'TICK':
            result.append(instruction)
            tick_count += 1
            # Add idle noise (p/10) at the start of each round
            if first_reset_seen and tick_count >= 2 and (tick_count - 1) % 9 == 1:
                result.append('DEPOLARIZE1', data_qubits, probability / 10)
                result.append('TICK')
                tick_count += 1
        else:
            result.append(instruction)

    return result


def bravyi_noise_model(
        circuit: stim.Circuit,
        qcode_res: dict,
        error_rate: float
) -> stim.Circuit:
    """
    Applies the Bravyi noise model to a noiseless Stim circuit.

    This noise model follows the BivariateBicycleCodes approach where:
    - All operations use the same base error_rate
    - Different gate types have different coefficients:
      * PrepX (RX): Z_ERROR with error_rate × 1
      * PrepZ (RZ): X_ERROR with error_rate × 1
      * MeasX (MX): Z_ERROR with error_rate × 1
      * MeasZ (MZ): X_ERROR with error_rate × 1
      * IDLE (I): DEPOLARIZE1 with error_rate × 2/3
      * CNOT: DEPOLARIZE2 with error_rate × 4/15 (for each of 3 error types)

    For CNOT gates, errors are decomposed into:
    - Control qubit only: 4/15
    - Target qubit only: 4/15
    - Both qubits: 4/15
    Total CNOT error probability: 3 × 4/15 = 4/5 of error_rate

    Args:
        circuit: The input noiseless quantum circuit
        qcode_res: Dictionary containing code information (to identify data qubits)
        error_rate: Base error probability

    Returns:
        stim.Circuit: A new circuit with noise operations inserted
    """
    from .circuit import create_qubit_registry

    # Get data qubit information
    Hx_positions = qcode_res['Hx_positions']
    Hz_positions = qcode_res['Hz_positions']
    Lx = qcode_res['Lx']
    Ly = qcode_res['Ly']
    grafted_qubits = qcode_res['grafted_qubits']

    qubit_registry = create_qubit_registry(Hx_positions, Hz_positions, Lx, Ly, grafted_qubits)
    DataL_keys = [q for q, (name, _, _) in qubit_registry.items() if name == "DataL"]
    DataR_keys = [q for q, (name, _, _) in qubit_registry.items() if name == "DataR"]
    XCheck_keys = [q for q, (name, _, _) in qubit_registry.items() if name == "XCheck"]
    ZCheck_keys = [q for q, (name, _, _) in qubit_registry.items() if name == "ZCheck"]
    data_qubits = DataL_keys + DataR_keys
    all_qubits = data_qubits + XCheck_keys + ZCheck_keys

    # Error rates for different operations (all same in Bravyi model)
    error_rate_init = error_rate
    error_rate_idle = error_rate
    error_rate_cnot = error_rate
    error_rate_meas = error_rate

    result = stim.Circuit()

    # State tracking
    first_reset_seen = False

    for instruction in circuit:
        # Handle repeat blocks recursively
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(
                repeat_count=instruction.repeat_count,
                body=bravyi_noise_model(
                    instruction.body_copy(),
                    qcode_res,
                    error_rate
                )))
        # Add X_ERROR after initial R (reset) gates on data qubits
        elif instruction.name == 'R' and not first_reset_seen:
            result.append(instruction)
            # For Z-basis reset, X_ERROR represents bit-flip
            result.append('X_ERROR', instruction.targets_copy(), error_rate_init)
            first_reset_seen = True

        # PrepX (RX): X-basis reset → Z_ERROR
        elif instruction.name == 'RX':
            result.append(instruction)
            result.append('Z_ERROR', instruction.targets_copy(), error_rate_init)

        # PrepZ (RZ): Z-basis reset → X_ERROR
        elif instruction.name == 'RZ':
            result.append(instruction)
            result.append('X_ERROR', instruction.targets_copy(), error_rate_init)

        # MeasX (MX): X-basis measurement → Z_ERROR (before measurement)
        elif instruction.name == 'MX':
            result.append('Z_ERROR', instruction.targets_copy(), error_rate_meas)
            result.append(instruction)

        # MeasZ (MZ): Z-basis measurement → X_ERROR (before measurement)
        elif instruction.name == 'MZ':
            result.append('X_ERROR', instruction.targets_copy(), error_rate_meas)
            result.append(instruction)

        # IDLE (I): DEPOLARIZE1
        # DEPOLARIZE1(p) applies X, Y, Z each with probability p/3
        # BivariateBicycleCodes uses single Pauli errors (Z or X) with prob 2/3 × error_rate
        # But DEPOLARIZE1 distributes to all 3 Paulis, so we use error_rate directly
        elif instruction.name == 'I':
            result.append(instruction)
            result.append('DEPOLARIZE1', instruction.targets_copy(), error_rate_idle)

        # CNOT (or CX, CZ): DEPOLARIZE2
        # DEPOLARIZE2(p) applies 15 two-qubit Pauli errors uniformly
        # BivariateBicycleCodes uses specific Pauli errors (IX, XI, XX or IZ, ZI, ZZ)
        # each with prob 4/15 × error_rate, but DEPOLARIZE2 distributes uniformly
        elif instruction.name in ['CNOT', 'CX', 'CZ']:
            result.append(instruction)
            result.append('DEPOLARIZE2', instruction.targets_copy(), error_rate_cnot)

        # Legacy support for MR (measure-reset) gates
        elif instruction.name == 'MR':
            # Before measurement: represents measurement error
            result.append('X_ERROR', instruction.targets_copy(), error_rate_meas)
            result.append(instruction)
            # After reset: represents reset error
            result.append('X_ERROR', instruction.targets_copy(), error_rate_init)

        # Legacy support for M (measure) gates
        elif instruction.name == 'M':
            result.append('X_ERROR', instruction.targets_copy(), error_rate_meas)
            result.append(instruction)

        # All other instructions (TICK, QUBIT_COORDS, DETECTOR, OBSERVABLE_INCLUDE, etc.)
        else:
            result.append(instruction)

    return result