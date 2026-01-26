import torch
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.engine import Engine
from tneq_qc.optim.optimizer import Optimizer

from img_recon import ImgProcess

# 梯度计算函数
def contract_for_decomposition_gradient(engine: Engine, 
                                        qctn: QCTN, 
                                        target_tensor):
    """
    修正版：专门用于张量分解的梯度计算 (单层收缩)。
    """
    # 1. 准备 Target
    target = engine.backend.convert_to_tensor(target_tensor)
    
    # 2. 获取/编译“纯 Core”收缩表达式 (单层)
    # 我们利用 engine.contractor (EinsumStrategy) 的底层能力
    cache_key = '_expr_core_only'
    
    if not hasattr(qctn, cache_key):
        # 这一步会自动分析 qctn 的拓扑，生成只把 Core 连起来的 einsum 公式
        einsum_eq, tensor_shapes = engine.contractor.build_core_only_expression(qctn)
        print(f"[Train] Compiled core-only expression: {einsum_eq}")
        # 编译优化后的表达式
        compute_fn = engine.contractor.create_contract_expression(einsum_eq, tensor_shapes)
        setattr(qctn, cache_key, compute_fn)
        print(f"[Train] Compiled core-only expression: {einsum_eq}")
    else:
        compute_fn = getattr(qctn, cache_key)

    # 3. 准备参数 (提取 TNTensor 的 tensor 部分，因为我们直接学习数值)
    raw_core_tensors = []
    for c_name in qctn.cores:
        c = qctn.cores_weights[c_name]
        # 如果是 TNTensor，取出的 .tensor 也是带梯度的
        if hasattr(c, 'tensor'):
            raw_core_tensors.append(c.tensor)
        else:
            raw_core_tensors.append(c)

    # 4. 定义 Loss 函数
    def mse_loss_fn(*core_tensors_args):
        # 执行收缩
        pred_tensor = engine.backend.execute_expression(compute_fn, *core_tensors_args)
        
        # 【关键修改】 强制 Reshape
        # 只要元素总数一致 (numel)，view/reshape 就能成功
        if pred_tensor.numel() != target.numel():
             raise ValueError(f"元素数量不匹配! 预测: {pred_tensor.numel()} (2^{len(pred_tensor.shape)}), "
                              f"目标: {target.numel()}. 请检查 n_cores 是否等于 log2(H*W)")
            
        pred_tensor = pred_tensor.reshape(target.shape)
        
        # 计算 MSE
        diff = pred_tensor - target
        loss = (diff * diff).mean()
        return loss

    # 5. 计算梯度
    argnums = list(range(len(raw_core_tensors)))
    value_and_grad_fn = engine.backend.compute_value_and_grad(mse_loss_fn, argnums=argnums)
    
    loss, grads = value_and_grad_fn(*raw_core_tensors)
    
    return loss, grads

def loop(engine: Engine, 
         qctn: QCTN, 
         optimizer: Optimizer, 
         target_tensor,
         Processor: ImgProcess
         ):
    
    while optimizer.iter < optimizer.max_iter:
        loss, grads = contract_for_decomposition_gradient(engine, qctn, target_tensor)

        loss_value = loss.item() if hasattr(loss, 'item') else float(loss)
        
        optimizer._apply_lr_schedule()

        if optimizer.tol and loss_value < optimizer.tol:
                print(f"Convergence achieved at iteration {optimizer.iter} with loss {loss_value}.")
                break
        
        print(f"Iteration {optimizer.iter}: loss = {loss_value} lr = {optimizer.learning_rate}")
        
        optimizer.step(qctn, grads)
        optimizer.iter += 1
    else:
        print(f"Maximum iterations reached: {optimizer.max_iter} with final loss {loss_value}.")
