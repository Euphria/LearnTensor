import torch

def generate_circuit_states_list(num_qubits, K, device='cuda'):
    """
    Generate circuit states list with status [0, 0, ..., 1] for each qubit
    Parameters:
    - num_qubits: Number of qubits
    - K: Dimension of each qubit state
    Returns:
    - circuit_states_list: List of tensors representing the circuit states for each qubit
    """
    circuit_states_list = [torch.zeros(K, device=device) for _ in range(num_qubits)]

    for i in range(len(circuit_states_list)):
        circuit_states_list[i][-1] = 1.0

    return circuit_states_list

if __name__ == "__main__":
    # 测试代码
    num_qubits = 3
    K = 3
    device = 'cuda'
    states_list = generate_circuit_states_list(num_qubits, K, device)
    for i, state in enumerate(states_list):
        print(f"Qubit {i} state:\n{state}\n")