import h5py
import numpy as np
import os
import torch
from datamodules.data_preprocess import QuantumHermitePreprocessor

def load_data(data_path: str,
              n_qubits: int,
              K: int,
              batch_size: int = 64,
              device: str = 'cuda') -> dict:
    """
    加载 .h5 格式的数据并返回train_data
    
    :param data_path: 数据路径
    :return: 加载的数据，形状为 B * T * N
    """
    if not os.path.exists(data_path):
        raise ValueError(f"找不到数据文件: {data_path}")

    # 使用 h5py 打开文件
    with h5py.File(data_path, 'r') as f:
        # 查看文件中的键 (Keys)
        keys = list(f.keys())
        # print(f"H5 文件包含的键: \t{keys}")
        
        if 'train_data' in f:
            train_data = f['train_data'][:]
        else:
            raise ValueError(f"{data_path}中没有 train_data 键")

    print(f"[Data] 成功加载数据文件：\t{os.path.abspath(data_path)}") # 应该是 B * T * N
    print(f"[Data] 数据形状 (B * T * N) : \t{train_data.shape}")

    # 由（B, T, N）转换为 (B*T, N) 的形式以便后续处理
    total_trials = train_data.shape[0] * train_data.shape[1]
    x_flattened = train_data.reshape(total_trials, -1)
    x = torch.from_numpy(x_flattened).float()

    processor = QuantumHermitePreprocessor(n_qubits=n_qubits, K=K, device=device)

    data_list = []
    num_batches = total_trials // batch_size
    
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_x = x[start:end]
        print(f"[Data] 处理第 {i+1}/{num_batches} 个批次，数据范围: [{start}:{end}]")
        
        # 得到包含 Mx_list 的字典
        batch_data = processor.process(batch_x)
        data_list.append(batch_data)
    
    if len(data_list) > 0:
        # 获取第一个 Batch 的第一个 Qubit 的 Mx 形状
        # 结构：data_list[batch_idx]['measure_input_list'][qubit_idx] -> Tensor(batch_size, K, K)
        mx_shape = data_list[0]['measure_input_list'][0].shape
        last_mx_shape = data_list[-1]['measure_input_list'][0].shape
        # phi_shape = data_list[0]['phi_x'].shape
        
        print(f"[Data] 预处理完成。")
        print(f"[Data] data_list 长度 (Batch 数量):  {len(data_list)}")
        print(f"[Data] 每个 Batch 的 Mx 形状 (BatchSize, K, K):  {list(mx_shape)}")
        print(f"[Data] 最后 Batch 的 Mx 形状 (BatchSize, K, K):  {list(last_mx_shape)}")
        # print(f"[Data] 每个 Batch 的 phi_x 形状 (BatchSize, Qubits, K): {list(phi_shape)}")

    return data_list

if __name__ == "__main__":
    # 你提供的测试代码
    data_path = "data/data_10_10_3.h5"
    
    # 为了防止演示时报错，如果没有文件则创建一个模拟文件
    if not os.path.exists(data_path):
        os.makedirs('data', exist_ok=True)
        with h5py.File(data_path, 'w') as f:
            f.create_dataset('data', data=np.random.randn(10, 10, 3))
            
    test_data = load_data(data_path, n_qubits=3, K=3, batch_size=32, device='cuda')
    for i, batch in enumerate(test_data):
        print(f"批次 {i+1} - measure_input_list 长度: {len(batch['measure_input_list'])}")
        for j, tensor in enumerate(batch['measure_input_list']):
            print(f"  Qubit {j}: {tensor.shape}")
        print(f"  phi_x 形状: {batch['phi_x'].shape}")