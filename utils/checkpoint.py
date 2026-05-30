"""Checkpoint 管理：meta-init / pretrained-DQfD / finetuned / best-eval 四类。"""
import os
import torch


def save_checkpoint(state: dict, path: str, ckpt_type: str = "meta"):
    """保存 checkpoint。"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({"type": ckpt_type, **state}, path)


def load_checkpoint(path: str, device: torch.device = None) -> dict:
    """加载 checkpoint。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device or "cpu")
    return ckpt


def get_checkpoint_path(base_dir: str, ckpt_type: str, iteration: int = None) -> str:
    """获取 checkpoint 路径。"""
    os.makedirs(base_dir, exist_ok=True)
    if iteration is not None:
        return os.path.join(base_dir, f"{ckpt_type}_iter{iteration:06d}.pt")
    return os.path.join(base_dir, f"{ckpt_type}.pt")
