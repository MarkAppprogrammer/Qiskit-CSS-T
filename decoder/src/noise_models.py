import stim

def standard_depolarizing_noise_model(
        circuit: stim.Circuit,
        data_qubits: list,
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
        elif instruction.name == 'R':
            result.append(instruction)
            if first_reset_seen:
                result.append('X_ERROR', instruction.targets_copy(), after_reset_flip_probability)
            first_reset_seen = True
        elif instruction.name == 'CNOT' or instruction.name == 'CX':
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
            # Apply data depolarization at the start of a round (heuristic: after first reset)
            if first_reset_seen:
                result.append('DEPOLARIZE1', data_qubits, before_round_data_depolarization)
        else:
            result.append(instruction)

    return result


def si1000_noise_model(
        circuit: stim.Circuit,
        data_qubits: list,
        all_qubits: list,
        probability: float
) -> stim.Circuit:

    result = stim.Circuit()
    first_reset_seen = False
    
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(
                repeat_count=instruction.repeat_count,
                body=si1000_noise_model(
                    instruction.body_copy(),
                    data_qubits,
                    all_qubits,
                    probability
                )))
        elif instruction.name == 'R':
            result.append(instruction)
            if first_reset_seen:
                result.append('X_ERROR', instruction.targets_copy(), 2 * probability)
                # Idle noise on non-reset qubits
                targets = set(t.value for t in instruction.targets_copy())
                idle_qubits = [q for q in all_qubits if q not in targets]
                if idle_qubits:
                    result.append('DEPOLARIZE1', idle_qubits, 2 * probability)
            first_reset_seen = True
            
        elif instruction.name in ['CNOT', 'CX']:
            result.append(instruction)
            result.append('DEPOLARIZE2', instruction.targets_copy(), probability / 10)
            
        elif instruction.name == 'MR':
            # Pre-measurement noise
            result.append('DEPOLARIZE1', instruction.targets_copy(), probability)
            result.append('X_ERROR', instruction.targets_copy(), 5 * probability)
            result.append(instruction)
            # Post-reset noise
            result.append('X_ERROR', instruction.targets_copy(), 2 * probability)
            
            targets = set(t.value for t in instruction.targets_copy())
            idle_qubits = [q for q in all_qubits if q not in targets]
            if idle_qubits:
                result.append('DEPOLARIZE1', idle_qubits, 2 * probability)
                
        elif instruction.name == 'M':
            result.append('DEPOLARIZE1', instruction.targets_copy(), probability)
            result.append('X_ERROR', instruction.targets_copy(), 5 * probability)
            result.append(instruction)
            
        elif instruction.name == 'TICK':
            result.append(instruction)
            # Idle noise at start of round
            if first_reset_seen:
                 result.append('DEPOLARIZE1', data_qubits, probability / 10)
        else:
            result.append(instruction)

    return result


def bravyi_noise_model(
        circuit: stim.Circuit,
        error_rate: float
) -> stim.Circuit:

    error_rate_init = error_rate
    error_rate_idle = error_rate
    error_rate_cnot = error_rate
    error_rate_meas = error_rate

    result = stim.Circuit()
    first_reset_seen = False

    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(
                repeat_count=instruction.repeat_count,
                body=bravyi_noise_model(instruction.body_copy(), error_rate)
            ))
        elif instruction.name == 'R':
            result.append(instruction)
            if first_reset_seen:
                result.append('X_ERROR', instruction.targets_copy(), error_rate_init)
            first_reset_seen = True
        elif instruction.name == 'RX':
            result.append(instruction)
            result.append('Z_ERROR', instruction.targets_copy(), error_rate_init)
        elif instruction.name == 'RZ':
            result.append(instruction)
            result.append('X_ERROR', instruction.targets_copy(), error_rate_init)
        elif instruction.name == 'MX':
            result.append('Z_ERROR', instruction.targets_copy(), error_rate_meas)
            result.append(instruction)
        elif instruction.name == 'MZ':
            result.append('X_ERROR', instruction.targets_copy(), error_rate_meas)
            result.append(instruction)
        elif instruction.name == 'I':
            result.append(instruction)
            result.append('DEPOLARIZE1', instruction.targets_copy(), error_rate_idle)
        elif instruction.name == 'TICK':
            result.append(instruction)
        elif instruction.name in ['CNOT', 'CX', 'CZ']:
            result.append(instruction)
            result.append('DEPOLARIZE2', instruction.targets_copy(), error_rate_cnot)
        elif instruction.name == 'MR':
            result.append('X_ERROR', instruction.targets_copy(), error_rate_meas)
            result.append(instruction)
            result.append('X_ERROR', instruction.targets_copy(), error_rate_init)
        elif instruction.name == 'M':
            result.append('X_ERROR', instruction.targets_copy(), error_rate_meas)
            result.append(instruction)
        else:
            result.append(instruction)

    return result