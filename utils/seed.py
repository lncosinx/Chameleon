"""
全局随机种子管理。
固定 torch / numpy / random 种子，保证可复现。
"""
import random
import os
import numpy as np
import torch


def set_seed(seed: int, deterministic_cudnn: bool = False):
    """
    设置全局随机种子。

    Args:
        seed: 随机种子。
        deterministic_cudnn: 是否启用 cudnn deterministic 模式
                           （开启后略慢，但可复现卷积操作）。
                           默认 False，训练时推荐 True。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark 模式可自动选最优 conv 算法（非 deterministic 但更快）
        torch.backends.cudnn.benchmark = True

    # Python hash 也要固定（影响某些数据结构迭代序）
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id: int):
    """
    DataLoader worker 的初始化函数。
    每个 worker 用 base_seed + worker_id 派生独立种子，
    避免全部 worker 产出一致数据。
    """
    base_seed = torch.initial_seed() % (2 ** 31)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
