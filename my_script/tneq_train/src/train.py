import sys
import os
import torch
import shutil
from pathlib import Path
from datetime import datetime
import  time
import argparse

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取当前脚本的父目录（即 src）
parent_dir = os.path.dirname(current_dir)
# 将项目根目录添加到系统路径中
sys.path.append(parent_dir)

# 自己写的模块
from load_config import load_config
from datamodules import load_data
from circuit_states import generate_circuit_states_list
from post_processing import fingure_loss, run_sample



#  引入 TNEQ 模块
from tneq_qc.core.engine_siamese import EngineSiamese
from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN, QCTNHelper
from tneq_qc.optim.optimizer import Optimizer


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # 这里的 flush 是为了兼容 Python 的输出缓冲机制
        self.terminal.flush()
        self.log.flush()

def main(device: str = 'cuda',
         save_path: str = './assets/') -> None:
    ####################################################### 
    # 整理执行思路是：
    # 1. 加载配置文件
    # 2. 加载数据：数据预处理
    # 3. 初始化量子状态向量
    # 4. 设计量子电路
    # 5. 训练模型
    # 6. 保存模型
    # 7. 保存日志
    ####################################################### 
    parser = argparse.ArgumentParser(description="TNEQ Quantum Training Script")
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='Path to the config file')

    args = parser.parse_args()

    # 一些初始化
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    #  日志初始化
    log_path =  "output.log"
    sys.stdout = Logger(str(log_path), sys.stdout)
    sys.stderr = Logger(str(log_path), sys.stderr)

    #######################################################  
    # 1. 加载配置文件
    #######################################################
    print("\n" + "="*50)
    print("加载配置文件...")
    print("="*50)

    config = load_config(args.config)

    #######################################################  
    # 2. 加载数据：数据预处理
    #######################################################
    print("\n" + "="*50)
    print("加载数据...")
    print("="*50)
    train_data = load_data(data_path=config['data_path'],
                           n_qubits=config['model_params']['n_qubits'],
                           K=config['model_params']['k_max'],
                           batch_size=config['batch_size'],
                           device=device)

    #######################################################  
    # 3. 初始化量子状态向量
    #######################################################
    print("\n" + "="*50)
    print("初始化量子电路状态...")
    print("="*50)
    backend = BackendFactory.create_backend(config['backend_type'], device=device)
    engine = EngineSiamese(backend=backend, strategy_mode=config['strategy_mode'])

    circuit_states_list = generate_circuit_states_list(num_qubits=config['model_params']['n_qubits'],
                                                       K=config['model_params']['k_max'],
                                                       device=device)

    ####################################################### 
    # 4. 设计量子电路
    ####################################################### 
    print("\n" + "="*50)
    print("量子电路图")
    print("="*50)

    qctn_graph = config['model_params']['qctn_graph']
    qctn = QCTN(qctn_graph, backend=engine.backend)
    print("量子电路结构：")
    print(f'{qctn.graph}')

    ####################################################### 
    # 5. 训练模型
    ####################################################### 
    print("\n" + "="*50)
    print("训练模型...")
    print("="*50)
    
    print("开始训练...")
    optimizer = Optimizer(method=config['training_params']['method'],
                          learning_rate=config['training_params']['lr_rate'],
                          max_iter=config['training_params']['max_iter'],
                          tol=config['training_params']['tol'],
                          beta1=config['training_params']['beta1'],
                          beta2=config['training_params']['beta2'],
                          epsilon=config['training_params']['epsilon'],
                          engine=engine,
                          lr_schedule= config['training_params']['lr_schedule'],
                          momentum=config['training_params']['momentum'],
                          stiefel=config['training_params']['stiefel'])
    
    torch.cuda.empty_cache()

    
    tic = time.time()

    optimizer.optimize(qctn, 
                       data_list=train_data, 
                    #    circuit_states=circuit_states_list,
                       circuit_states_list=circuit_states_list,
                       )

    toc = time.time()
    
    print(f"已分配显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"缓存显存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Optimization Time: {toc - tic:.2f} seconds")
    print("训练完成。")

    ####################################################### 
    # 6. 保存模型
    ####################################################### 
    print("\n" + "="*50)
    print("保存模型...")
    print("="*50)

    # 保存路径： assets/dataset_name/timestamp
    dataset_name = Path(config['data_path']).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = Path(save_path) / dataset_name / timestamp

    if save_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M_%S")  # 添加秒数
        save_dir = Path(save_path) / dataset_name / timestamp

    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Save] 保存路径: {save_dir}")

    # 保存配置文件
    shutil.copy(args.config, save_dir)
    print(f"[Save] 配置文件已保存。")

    # 保存数据
    shutil.copy(config['data_path'], save_dir)
    print(f"[Save] 数据文件已保存。")

    # 保存模型
    cores_save_path = save_dir / "qctn_cores.safetensors"
    metadata_dict = {
        "backend_type":     config['backend_type'],
        "strategy_mode":    config['strategy_mode'],
        "qctn_graph":       qctn_graph, 
        "n_qubits":         config['model_params']['n_qubits'],
        "k_max":            config['model_params']['k_max'],
        "dataset":          dataset_name,
        "timestamp":        timestamp
    }
    qctn.save_cores(str(cores_save_path), metadata=metadata_dict)
    print(f'\nmetadata keys:')
    for key in metadata_dict:
        print(f' - {key} ')
    print(f"[Save] 模型已保存。")

    ####################################################### 
    # 7. 结果处理
    ####################################################### 

    # 写入日志文件
    sys.stdout.log.flush()
    sys.stderr.log.flush()

    print("\n" + "="*50)
    print("结果处理...")
    print("="*50)
    
    # 生成 Loss 曲线图
    if config['post_processing_params']['plot_loss']:
        try:
            fingure_loss(save_dir, log_path)
        except Exception as e:
            print(f"[Post Error] 生成 Loss 曲线图时出错: {e}")
    
    # 采样
    if config['post_processing_params']['sample']:
        print("\n开始采样...")
        run_sample(file_path=cores_save_path, 
                   num_samples=config['post_processing_params']['num_samples'], 
                   device=device)
        
        print("\n采样完成。")

    #######################################################
    # 保存日志
    #######################################################
    
    # 关闭日志文件句柄
    sys.stdout.log.close()
    sys.stderr.log.close()

    # 恢复原始 stdout 和 stderr
    sys.stdout = sys.stdout.terminal
    sys.stderr = sys.stderr.terminal
    shutil.move(log_path, save_dir)
    pass

if __name__ == '__main__':
    main()