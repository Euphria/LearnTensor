##############################################
#  Training the model
##############################################

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
from img_recon import ImgProcess
from loop import loop
from process import fingure_loss

#  引入 TNEQ 模块
from tneq_qc.core.engine import Engine
from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN, QCTNHelper
from tneq_qc.optim.optimizer import Optimizer
from tneq_qc.core.tn_tensor import TNTensor

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


# 生成core graph
def create_core_graph(n_cores: int, 
                      bond_dim: int) -> str:
    # 生成 A, B, C... 这样的独立 Core 名字
    core_names = []
    for i in range(n_cores):
        if i < 26:  # A-Z
            core_names.append(chr(ord('A') + i))
        else:  # 如果超过26个核心，使用AA, AB, AC...等格式
            first_char = chr(ord('A') + (i // 26) - 1)
            second_char = chr(ord('A') + (i % 26))
            core_names.append(first_char + second_char)
    
    graph_lines = []
    for i, name in enumerate(core_names):
        # 格式: -Bond_Left- Name -Bond_Right-
        # 首尾 Bond 为 1 (或者 tneq 里的 2 实际上是 rank)
        # 这里用 tneq 的简写格式 -rank-Name-rank-
        graph_lines.append(f"-{bond_dim}-{name}-{bond_dim}-")
    
    graph_str = "\n".join(graph_lines)
    print("Generated core graph:")
    print(graph_str)
    return graph_str
# def create_core_graph(n_cores: int, bond_dim: int) -> str:
#     """
#     生成标准的 MPS (Zipper结构) Graph。
#     解决显存爆炸问题，确保 Core 之间有 Bond 连接。
#     """
#     if n_cores <= 0: return ""
    
#     # 生成名字 A..Z..
#     core_names = []
#     for i in range(n_cores):
#         if i < 26: core_names.append(chr(ord('A') + i))
#         else: core_names.append(chr(ord('A') + (i // 26) - 1) + chr(ord('A') + (i % 26)))

#     # 定义维度字符串
#     # 物理腿 dim=2 (对应二进制像素), Bond腿 dim=bond_dim
#     phy_str = "-2-" 
#     bond_str = f"-{bond_dim}-"
    
#     # 占位符 (用于对齐)
#     empty_phy = "-" * len(phy_str)
#     empty_bond = "-" * len(bond_str)
#     empty_name = "-"

#     graph_lines = []
    
#     # 我们构建一个 "之" 字形或者简单的链式结构
#     # 既然 opt_einsum 很聪明，我们只需要明确写出连接关系即可
#     # 为了视觉清晰，我们生成如下结构:
#     # Row 0: -2-A-2-            (Physical A)
#     # Row 1:    -8-A-8-B-8-     (Bond A-B)
#     # Row 2:       -2-B-2-      (Physical B)
#     # Row 3:          -8-B-8-C- (Bond B-C)
    
#     for i in range(n_cores):
#         # 1. 物理层 (Physical Line)
#         # 缩进: 前面有 i 个 Bond 块的长度
#         indent = (empty_bond + empty_name) * i
#         line_phy = f"{indent}{phy_str}{core_names[i]}{phy_str}"
#         graph_lines.append(line_phy)
        
#         # 2. 连接层 (Bond Line) - 最后一个 Core 后面不需要 Bond
#         if i < n_cores - 1:
#             # 缩进: 前面有 i 个 Bond 块 + 半个偏移? 
#             # 其实只要这行写了 A 和 B，opt_einsum 就能认出来。对齐只是为了好看。
#             # 我们简单地把 A 和 B 连起来
#             indent_bond = (empty_bond + empty_name) * i
#             # 格式: -8-A-8-B-
#             line_bond = f"{indent_bond}   {bond_str}{core_names[i]}{bond_str}{core_names[i+1]}"
#             graph_lines.append(line_bond)

#     graph_str = "\n".join(graph_lines)
#     print(f"\n[Graph] Generated MPS Graph (Zipper Chain):")
#     print(graph_str)
#     return graph_str



def main(device: str = 'cuda') -> None:
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
    # 1. 初始化
    #######################################################
    print("\n" + "="*50)
    print("初始化...")
    print("="*50)

    config = load_config(args.config)
    
    # 保存路径： assets/dataset_name/timestamp
    dataset_name = Path(config['data_path']).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = Path(config['save_params']['output_dir']) / dataset_name / timestamp

    if save_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M_%S")  # 添加秒数
        save_dir = Path(config['save_params']['output_dir']) / dataset_name / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Save] 保存路径: {save_dir}")

    # 初始化 backend 和 engine
    backend = BackendFactory.create_backend(config['backend_type'], device=device)
    engine = Engine(backend=backend, strategy_mode=config['strategy_mode'])

    #######################################################  
    # 2. 加载数据：数据预处理
    #######################################################
    print("\n" + "="*50)
    print("加载数据...")
    print("="*50)
    Processor = ImgProcess(img_path=config['data_path'],
                              output_dir=save_dir,
                              target_size=config['model_params']['target_size'])
    
    target_matrix = Processor.img_to_matrix()
    print(f'target_metric: {target_matrix}')

    ####################################################### 
    # 4. 设计core graph
    ####################################################### 
    print("\n" + "="*50)
    print("初始化core graph...")
    print("="*50)
    
    if config['model_params'].get('auto_graph', True):
        qctn_graph = create_core_graph(config['model_params']['n_cores'], 
                                   config['model_params']['BOND_DIM'])
    else:
        qctn_graph = config['model_params']['qctn_graph']
        
    qctn = QCTN(qctn_graph, backend=engine.backend)

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

    loop(engine, qctn, optimizer, target_matrix, Processor)


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

    # 保存配置文件
    shutil.copy(args.config, save_dir)
    print(f"[Save] 配置文件已保存。")

    # 保存模型
    cores_save_path = save_dir / "qctn_cores.safetensors"
    metadata_dict = {
        "backend_type":     config['backend_type'],
        "strategy_mode":    config['strategy_mode'],
        "qctn_graph":       qctn_graph, 
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
    
    # 生成 recon 图
    try:
        compute_fn = getattr(qctn, '_expr_core_only')
        raw_core_tensors = []
        for c_name in qctn.cores:
            c = qctn.cores_weights[c_name]
            # 如果是 TNTensor，取出的 .tensor 也是带梯度的
            if hasattr(c, 'tensor'):
                raw_core_tensors.append(c.tensor)
            else:
                raw_core_tensors.append(c)

        pred_tensor = engine.backend.execute_expression(compute_fn, *raw_core_tensors)
        pred_tensor = pred_tensor.reshape(target_matrix.shape)
        Processor.matrix_to_img(
            matrix=pred_tensor,
            save_name="recon.png"
        )
    except Exception as e:
        print(f"[Post Error] 生成重建图像时出错: {e}")

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