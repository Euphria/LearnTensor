import torch
import math
import numpy as np

class QuantumHermitePreprocessor:
    """
    基于 Hermite 多项式的量子数据预处理 (源于原数据预处理文件改的)
    """
    def __init__(self, n_qubits: int,
                 K: int, 
                 device='cuda') -> None:
        self.n_qubits = n_qubits
        self.K = K  # Hermite 多项式的阶数
        self.device = device
        # 初始化权重 (log factorial)
        self.weights = self._init_normalization_factors_vectorized(K, device)
        # 调整权重维度以便后续广播: (1, 1, K)
        self.weights = self.weights[None, None, :K]

    def _init_normalization_factors_vectorized(self, 
                                               k_max: int, 
                                               device: str) -> torch.Tensor:
        """计算归一化因子 sqrt(1/k!)"""
        k = torch.arange(k_max + 1, dtype=torch.float32)
        log_factorial = torch.lgamma(k + 1)
        log_2pi = math.log(2 * math.pi)
        log_factor = -0.5 * (0.5 * log_2pi + log_factorial)
        return torch.exp(log_factor).to(device)

    def _eval_hermitenorm_batch(self, n_max: int, 
                                x: torch.Tensor) -> torch.Tensor:
        """计算 Hermite 多项式"""
        # x shape: (Batch, D)
        x = x.to(self.device)
        H = torch.zeros((n_max + 1,) + x.shape, dtype=x.dtype, device=self.device)
        H[0] = torch.ones_like(x)
        
        if n_max >= 1:
            H[1] = x
            for i in range(2, n_max + 1):
                H[i] = x * H[i-1] - (i-1) * H[i-2]
        return H

    def process(self, x_input: torch.Tensor) -> dict:
        """
        核心处理管道
        输入 x_input: (Batch, n_qubits) 的原始数据 (假设已经归一化/或者是 raw)
        返回: data_dict 包含 'Mx_list' 和 'phi_x'
        """
        # 1. 确保输入是 Tensor 且在正确的设备上
        if not isinstance(x_input, torch.Tensor):
            x_input = torch.tensor(x_input, dtype=torch.float32, device=self.device)
        else:
            x_input = x_input.to(self.device)

        # 注意：原代码里有一个 uniform(-5, 5) 或者 normal(2.5, 1.0) 的生成过程
        # 如果你传入的是真实数据，你可能需要先把真实数据 Standardize (Z-score) 
        # 然后缩放到类似的区间，比如 [-3, 3] 以利用 Hermite 多项式的特性
        # 这里假设 x_input 已经是合适的分布
        
        # 2. 计算 Hermite 多项式
        # out shape: (K, B, D)
        out = self._eval_hermitenorm_batch(self.K - 1, x_input) 
        
        # 3. 维度变换 -> (B, D, K)
        out = out.permute(1, 2, 0).contiguous()

        # 4. 加权 (Gaussian kernel part)
        # phi(x) = weights * exp(-x^2/2) * H(x)
        out = self.weights * torch.sqrt(torch.exp(- torch.square(x_input) / 2))[:, :, None] * out

        # 5. 计算 Mx (Feature Matrix)
        # Mx = phi(x) * phi(x)^T (element-wise outer product along dimension K)
        # einsum "abc, abd -> abcd" 意味着:
        # a=Batch, b=Qubit(Dim), c=K, d=K -> 结果 (B, D, K, K)
        Mx = torch.einsum("abc,abd->abcd", out, out)

        # 6. 格式化输出 (为了适配 Optimizer)
        # 原代码返回的是一个 list of lists，我们需要保持一致
        # Mx_list 是一个 list，长度为 n_qubits，每个元素是 (Batch, K, K)
        Mx_list = [Mx[:, i] for i in range(self.n_qubits)]
        
        # 构造返回字典 (为了适配后面 contract 的接口)
        result = {
            "measure_input_list": Mx_list, # 用于训练时的 contraction
            # "phi_x": out                   # 原始特征向量，可能用于调试
        }
        
        return result