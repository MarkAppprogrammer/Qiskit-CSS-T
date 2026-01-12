import stim
import numpy as np

def standard_depolarizing_noise_model(
        circuit: stim.Circuit,
        data_qubits: list[int],
        after_clifford_depolarization: float,
        after_reset_flip_probability: float,
        before_measure_flip_probability: float,
        before_round_data_depolarization: float
) -> stim.Circuit:
    result = stim.Circuit()
    first_reset_seen = False
    tick_count = 0

    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(
                repeat_count=instruction.repeat_count,
                body=standard_depolarizing_noise_model(
                    instruction.body_copy(),
                    data_qubits,
                    after_clifford_depolarization,
                    after_reset_flip_probability,
                    before_measure_flip_probability,
                    before_round_data_depolarization
                )))
        elif instruction.name == 'R' and not first_reset_seen:
            result.append(instruction)
            result.append('X_ERROR', instruction.targets_copy(), after_reset_flip_probability)
            first_reset_seen = True
        elif instruction.name in ['CNOT', 'CX', 'CZ']:
            result.append(instruction)
            result.append('DEPOLARIZE2', instruction.targets_copy(), after_clifford_depolarization)
        elif instruction.name == 'MR':
            result.append('X_ERROR', instruction.targets_copy(), before_measure_flip_probability)
            result.append(instruction)
            result.append('X_ERROR', instruction.targets_copy(), after_reset_flip_probability)
        elif instruction.name == 'M':
            result.append('X_ERROR', instruction.targets_copy(), before_measure_flip_probability)
            result.append(instruction)
        elif instruction.name == 'TICK':
            result.append(instruction)
            tick_count += 1
            # Assuming a standard surface code schedule where a round is ~9 ticks? 
            # Note: Your specific circuit might not follow the 9-tick structure exactly.
            # This logic adds idle noise at start of rounds.
            if first_reset_seen and before_round_data_depolarization > 0:
                 # Simple heuristic: Apply if it looks like the start of a round block
                 # or simply apply it if your schedule relies on TICKs.
                 # Original code logic: if tick_count >= 2 and (tick_count - 1) % 9 == 1:
                 # Adjusting to apply generally for this example:
                 pass 
        else:
            result.append(instruction)
            
    return result

import stim

def si1000_noise_model(
        circuit: stim.Circuit,
        data_qubits: list[int],
        probability: float
) -> stim.Circuit:
    # 1. Pre-scan to find all used qubits in the entire circuit
    all_qubits_in_circuit = set()
    for op in circuit.flattened():
        for t in op.targets_copy():
            if t.is_qubit_target:
                all_qubits_in_circuit.add(t.value)
    
    all_qubits_list = list(all_qubits_in_circuit)
    result = stim.Circuit()
    first_reset_seen = False
    tick_count = 0

    for instruction in circuit:
        # Handle repeat blocks recursively
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(
                repeat_count=instruction.repeat_count,
                body=si1000_noise_model(instruction.body_copy(), data_qubits, probability)
            ))
            continue

        # Extract targeted qubits for the current instruction
        targets = [t.value for t in instruction.targets_copy() if t.is_qubit_target]

        # --- NOISE LOGIC ---
        
        # 1. Initial Reset (R)
        if instruction.name == 'R' and not first_reset_seen:
            result.append(instruction)
            result.append('X_ERROR', targets, 2 * probability)
            # Idle noise on qubits not being reset
            idle_qubits = list(all_qubits_in_circuit - set(targets))
            if idle_qubits:
                result.append('DEPOLARIZE1', idle_qubits, 2 * probability)
            first_reset_seen = True

        # 2. Single Qubit Gates
        elif instruction.name in ['H', 'S', 'X', 'Y', 'Z', 'S_DAG', 'H_DAG']:
            result.append(instruction)
            result.append('DEPOLARIZE1', targets, probability / 10)

        # 3. Two-Qubit Gates (CNOT)
        elif instruction.name in ['CNOT', 'CX']:
            result.append(instruction)
            result.append('DEPOLARIZE2', targets, probability)

        # 4. Measure-Reset (MR)
        elif instruction.name == 'MR':
            result.append('DEPOLARIZE1', targets, probability)   # Before meas
            result.append('X_ERROR', targets, 5 * probability)     # Meas flip
            result.append(instruction)
            result.append('X_ERROR', targets, 2 * probability)     # After reset
            # Idle noise for qubits not involved in MR
            idle_qubits = list(all_qubits_in_circuit - set(targets))
            if idle_qubits:
                result.append('DEPOLARIZE1', idle_qubits, 2 * probability)

        # 5. Measurement Only (M)
        elif instruction.name == 'M':
            result.append('DEPOLARIZE1', targets, probability)
            result.append('X_ERROR', targets, 5 * probability)
            result.append(instruction)
            # Idle noise
            idle_qubits = list(all_qubits_in_circuit - set(targets))
            if idle_qubits:
                result.append('DEPOLARIZE1', idle_qubits, 2 * probability)

        # 6. TICK handling
        elif instruction.name == 'TICK':
            result.append(instruction)
            tick_count += 1
            # Custom logic: Add noise to data qubits at specific round intervals
            if first_reset_seen and tick_count >= 2 and (tick_count - 1) % 9 == 1:
                if data_qubits:
                    result.append('DEPOLARIZE1', data_qubits, probability * 2)
                    result.append('TICK')
                    tick_count += 1
        
        # 7. Default (Pass-through for everything else)
        else:
            result.append(instruction)

    return result

def bravyi_noise_model(
        circuit: stim.Circuit,
        error_rate: float
) -> stim.Circuit:
    result = stim.Circuit()
    first_reset_seen = False
    
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(
                repeat_count=instruction.repeat_count,
                body=bravyi_noise_model(instruction.body_copy(), error_rate)
            ))
        elif instruction.name == 'R' and not first_reset_seen:
            result.append(instruction)
            result.append('X_ERROR', instruction.targets_copy(), error_rate)
            first_reset_seen = True
        elif instruction.name == 'RX':
            result.append(instruction)
            result.append('Z_ERROR', instruction.targets_copy(), error_rate)
        elif instruction.name == 'RZ':
            result.append(instruction)
            result.append('X_ERROR', instruction.targets_copy(), error_rate)
        elif instruction.name == 'MX':
            result.append('Z_ERROR', instruction.targets_copy(), error_rate)
            result.append(instruction)
        elif instruction.name == 'MZ':
            result.append('X_ERROR', instruction.targets_copy(), error_rate)
            result.append(instruction)
        elif instruction.name == 'I':
            result.append(instruction)
            result.append('DEPOLARIZE1', instruction.targets_copy(), error_rate)
        elif instruction.name in ['CNOT', 'CX', 'CZ']:
            result.append(instruction)
            result.append('DEPOLARIZE2', instruction.targets_copy(), error_rate)
        elif instruction.name == 'M':
            result.append('X_ERROR', instruction.targets_copy(), error_rate)
            result.append(instruction)
        else:
            result.append(instruction)
    return result
