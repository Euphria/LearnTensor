import torch
from pathlib import Path
import os
import sys

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取当前脚本的父目录（即 src）
parent_dir = os.path.dirname(current_dir)
# 将项目根目录添加到系统路径中
sys.path.append(parent_dir)

from src.load_config import load_config
from src.circuit_states import generate_circuit_states_list
from tneq_qc.core.engine_siamese import EngineSiamese
from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN

def run_sampling(model_dir, num_samples=10):
    model_dir = Path(model_dir)
    config = load_config(str(model_dir / "config.yaml"))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Sample] 正在使用设备: {device}")

    # 2. 初始化引擎和模型
    backend = BackendFactory.create_backend(config['backend_type'], device=device)
    engine = EngineSiamese(backend=backend, strategy_mode=config['strategy_mode'])
    
    # 这里的 graph 字符串必须与训练时一致
    qctn_graph = "-3-A-3-\n-3-A-3-\n-3-A-3-" 
    qctn = QCTN(qctn_graph, backend=backend)
    qctn.load_cores(str(model_dir / "qctn_cores.safetensors"))
    
    # 3. 准备初始电路状态 (通常是 |0> 态的变换)
    circuit_states = generate_circuit_states_list(
        num_qubits=config['model_params']['n_qubits'],
        K=config['model_params']['k_max'],
        device=device
    )

    # 4. 执行采样
    # 采样维度 dim 对应 Hermite 阶数 K
    print(f"[Sample] 开始生成 {num_samples} 个样本...")
    sampled_indices = engine.sample(
        qctn=qctn,
        circuit_states_list=circuit_states,
        num_samples=num_samples,
        K=config['model_params']['k_max']
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
    parser.add_argument('--dir', type=str, required=True, help='模型所在目录')
    parser.add_argument('--n', type=int, default=10, help='采样数量')
    args = parser.parse_args()

    run_sampling(args.dir, args.n)