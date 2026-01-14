import torch
from pathlib import Path
import os
import sys
from safetensors import safe_open

if __name__ == '__main__':
    import os
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取当前脚本的父目录（即 src）
    parent_dir = os.path.dirname(current_dir)
    # 将项目根目录添加到系统路径中
    sys.path.append(parent_dir)

from src.circuit_states import generate_circuit_states_list

from tneq_qc.core.qctn import QCTN
from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.engine_siamese import EngineSiamese


def run_sample(file_path, num_samples=10, device='cuda'):
    # 检查文件是否存在
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"[Error] 文件不存在: {file_path}")
        return

    # print("=" * 60)
    # print(f"正在检查文件: {file_path.absolute()}")
    # print("=" * 60)

    # 1. 加载模型配置
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
    except Exception as e:
        print(f"[Error] 读取文件元数据时出错: {e}")
        return
    
    # 2. 初始化引擎和模型

    backend = BackendFactory.create_backend(metadata['backend_type'], device=device)
    engine = EngineSiamese(backend=backend, strategy_mode=metadata['strategy_mode'])
    
    # 加载模型cores
    # print(f'qctn_graph: {metadata["qctn_graph"]}')
    qctn = QCTN(metadata['qctn_graph'], backend=backend)
    qctn.load_cores(file_path)
    
    # 3. 准备初始电路状态 (通常是 |0> 态的变换)
    circuit_states_list = generate_circuit_states_list(num_qubits=int(metadata['n_qubits']),
                                                       K=int(metadata['k_max']),
                                                       device=device)

    # 4. 执行采样
    # 采样维度 dim 对应 Hermite 阶数 K
    print(f"[Sample] 开始生成 {num_samples} 个样本...")
    sampled_indices = engine.sample(
        qctn=qctn,
        circuit_states_list=circuit_states_list,
        num_samples=num_samples,
        K=int(metadata['k_max'])
    )

    # 5. 结果显示
    print("\n" + "="*40)
    print(f"采样结果 (前 {num_samples} 个):")
    print("="*40)
    # sampled_indices 形状为 (num_samples, n_qubits)
    samples_np = sampled_indices.cpu().numpy()
    
    for i, sample in enumerate(samples_np):
        print(f"样本 {i+1}: 神经元状态组合 {sample}")
    
    return samples_np

if __name__ == "__main__":
    # 使用示例: python src/sample_model.py --dir assets/data_10_10_3/20260108_1605
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='模型地址')
    parser.add_argument('--n', type=int, default=10, help='采样数量')
    args = parser.parse_args()

    run_sample(args.file, args.n)