import torch
import numpy as np

def get_adam_avg_actual_lr(optimizer):
    """
    获取Adam优化器所有参数的平均实际学习率
    
    Adam实际更新公式：param = param - lr * m_hat / (sqrt(v_hat) + eps)
    其中：
    - lr: 基础学习率
    - m_hat = m / (1 - beta1^step): 偏差校正后的一阶矩
    - v_hat = v / (1 - beta2^step): 偏差校正后的二阶矩
    - step: 当前优化步数
    - v: 二阶矩估计（exp_avg_sq）
    - eps: 数值稳定性参数（默认1e-8）
    
    实际学习率 = lr / (sqrt(v_hat) + eps)，每个参数不同
    
    Args:
        optimizer: PyTorch优化器对象（Adam）
    
    Returns:
        float: 所有参数的实际学习率中位数（使用中位数避免训练初期异常值影响）
    """
    if len(optimizer.param_groups) == 0:
        return 0.0
    
    all_actual_lrs = []
    
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        beta2 = param_group.get('betas', (0.9, 0.999))[1]
        eps = param_group.get('eps', 1e-8)
        
        for param in param_group['params']:
            state = optimizer.state[param]
            if len(state) > 0 and 'exp_avg_sq' in state:
                # 已经执行过step，可以计算实际学习率
                step = state.get('step', 1)
                bias_correction2 = 1 - beta2 ** step
                v = state['exp_avg_sq']
                
                # Adam实际更新公式：param = param - lr * m_hat / (sqrt(v_hat) + eps)
                # v_hat = v / (1 - beta2^step)
                v_hat = v / (bias_correction2 + 1e-12)  # 偏差校正后的二阶矩，添加小值防止除零
                sqrt_v_hat = torch.sqrt(v_hat + eps)  # 添加eps保证数值稳定性
                actual_lr = lr / sqrt_v_hat
                
                # 收集所有参数的实际学习率
                if isinstance(actual_lr, torch.Tensor):
                    # 限制异常大的学习率（训练初期v_hat很小会导致学习率很大）
                    # 使用clamp限制在合理范围内，然后取中位数避免异常值
                    actual_lr_clamped = torch.clamp(actual_lr, min=lr * 0.01, max=lr * 100)
                    all_actual_lrs.append(actual_lr_clamped.median().item())
                else:
                    all_actual_lrs.append(min(max(actual_lr, lr * 0.01), lr * 100))
            else:
                # 还没有执行step，使用基础学习率
                all_actual_lrs.append(lr)
    
    if len(all_actual_lrs) == 0:
        return 0.0
    
    # 使用中位数而不是平均值，避免训练初期异常大的学习率影响
    return np.median(all_actual_lrs)

# import torch
import math

def get_effective_lr_stats(optimizer, module=None):
    """
    返回：
    - avg_eff_lr: 所有参数元素 effective lr 的平均值
    - max_eff_lr: 最大值
    - min_eff_lr: 最小值
    如果传入 module，则只统计该 module 的参数。
    """
    eff_lrs = []

    for group in optimizer.param_groups:
        lr = group['lr']
        beta1, beta2 = group['betas']
        eps = group['eps']

        for p in group['params']:
            if p.grad is None:
                continue
            if module is not None and p not in list(module.parameters()):
                continue

            state = optimizer.state[p]
            if len(state) == 0:
                # 还没更新过
                continue

            exp_avg = state['exp_avg']       # m_t
            exp_avg_sq = state['exp_avg_sq'] # v_t
            step = state['step']

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            m_hat = exp_avg / bias_correction1
            v_hat = exp_avg_sq / bias_correction2

            # effective lr（逐元素）
            eff_lr_tensor = lr / (torch.sqrt(v_hat) + eps)

            eff_lrs.append(eff_lr_tensor.detach().view(-1))

    if not eff_lrs:
        return None

    eff_lrs = torch.cat(eff_lrs)
    return {
        "avg": eff_lrs.mean().item(),
        "max": eff_lrs.max().item(),
        "min": eff_lrs.min().item(),
    }

def get_adam_step_size_stats(optimizer, module=None):
    """
    获取Adam优化器的实际参数更新量（step size）统计信息
    
    Adam实际更新公式：param = param - lr * m_hat / (sqrt(v_hat) + eps)
    实际更新量（step size）= lr * m_hat / (sqrt(v_hat) + eps)
    
    这个值比effective learning rate更有意义，因为它包含了m_hat的影响。
    在训练初期，虽然 lr / (sqrt(v_hat) + eps) 很大，但 m_hat 很小，
    所以实际更新量通常不会过大。
    
    Args:
        optimizer: PyTorch优化器对象（Adam）
        module: 可选，如果提供则只统计该module的参数
    
    Returns:
        dict: {
            "avg": 平均更新量,
            "max": 最大更新量,
            "min": 最小更新量,
            "avg_abs": 平均绝对更新量
        } 或 None（如果没有参数）
    """
    step_sizes = []
    
    for group in optimizer.param_groups:
        lr = group['lr']
        beta1, beta2 = group['betas']
        eps = group['eps']
        
        for p in group['params']:
            # if p.grad is None:
            #     continue
            if module is not None and p not in list(module.parameters()):
                continue
            
            state = optimizer.state[p]
            if len(state) == 0:
                continue
            
            exp_avg = state['exp_avg']       # m_t
            exp_avg_sq = state['exp_avg_sq'] # v_t
            step = state['step']
            
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            m_hat = exp_avg / (bias_correction1 + 1e-12)
            v_hat = exp_avg_sq / (bias_correction2 + 1e-12)
            
            # 实际更新量（step size）= lr * m_hat / (sqrt(v_hat) + eps)
            step_size = lr * m_hat / (torch.sqrt(v_hat) + eps)
            
            step_sizes.append(step_size.detach().view(-1))
    
    if not step_sizes:
        return None
    
    step_sizes = torch.cat(step_sizes)
    
    return step_sizes.abs().mean().item()
    return {
        "avg": step_sizes.mean().item(),
        "max": step_sizes.max().item(),
        "min": step_sizes.min().item(),
        "avg_abs": step_sizes.abs().mean().item(),  # 平均绝对更新量，更有参考价值
    }
