import sys
import os
import torch

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取当前脚本的父目录（即 src）
parent_dir = os.path.dirname(current_dir)
# 将项目根目录添加到系统路径中
sys.path.append(parent_dir)

from load_config import load_config
from datamodules.load_data import load_data
from circuit_states import generate_circuit_states_list
import argparse

from tneq_qc.core.engine_siamese import EngineSiamese
from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN, QCTNHelper
from tneq_qc.optim.optimizer import Optimizer

def main(device: str = 'cuda'):
    # 整理执行思路是：
    # 1. 加载配置文件
    # 2. 加载数据：数据预处理
    # 3. 训练模型
    # 4. 保存模型
    parser = argparse.ArgumentParser(description="TNEQ Quantum Training Script")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')

    args = parser.parse_args()

    # 一些初始化
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    backend = BackendFactory.create_backend('pytorch', device=device)
    engine = EngineSiamese(backend=backend, strategy_mode="balanced")

    # 1. 加载配置文件
    print("\n" + "="*50)
    print("加载配置文件...")
    print("="*50)

    config = load_config(args.config)

    # 2. 加载数据
    print("\n" + "="*50)
    print("加载数据...")
    print("="*50)
    train_data = load_data(data_path=config['data_path'],
                           n_qubits=config['model_params']['n_qubits'],
                           K=config['model_params']['k_max'],
                           batch_size=config['batch_size'],
                           device=device)

    # 3. 初始化量子状态向量
    print("\n" + "="*50)
    print("初始化量子状态向量...")
    print("="*50)
    circuit_states_list = generate_circuit_states_list(num_qubits=config['model_params']['n_qubits'],
                                                       K=config['model_params']['k_max'],
                                                       device=device)

    # 4. 设计量子电路
    print("\n" + "="*50)
    print("量子电路图")
    print("="*50)

    qctn_graph = "-3-A-3-\n" \
                "-3-A-3-\n" \
                "-3-A-3-"
    qctn = QCTN(qctn_graph, backend=engine.backend)

    # 5. 训练模型
    print("\n" + "="*50)
    print("训练模型...")
    print("="*50)

    # 6. 保存模型
    print("\n" + "="*50)
    print("保存模型...")
    print("="*50)

    pass

if __name__ == '__main__':
    main()