"""
共享内存 rollout 缓冲。

连续内存结构化数组，支持零拷贝跨进程传递。
"""
import numpy as np
from typing import Tuple, Optional


class SharedRolloutBuffer:
    """共享内存 rollout 缓冲（环形）。"""

    def __init__(self, max_episodes: int = 256, max_steps: int = 512,
                 num_agents: int = 8, K: int = 11, patch_channels: int = 4):
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.K = K
        self.patch_channels = patch_channels

        # 结构化数组；patch_channels=4（原生3ch + 足迹1ch，INVARIANT 5）
        self.obs_buffer = np.zeros(
            (max_episodes, max_steps, num_agents, patch_channels, K, K), dtype=np.uint8
        )
        self.action_buffer = np.zeros(
            (max_episodes, max_steps, num_agents), dtype=np.int8
        )
        self.reward_buffer = np.zeros(
            (max_episodes, max_steps, num_agents), dtype=np.float16
        )
        self.done_buffer = np.zeros(
            (max_episodes, max_steps), dtype=bool
        )
        self.mask_buffer = np.zeros(
            (max_episodes, max_steps, num_agents, 5), dtype=bool
        )
        self.version_buffer = np.zeros(max_episodes, dtype=np.int32)

        self._write_idx = 0
        self._count = 0

    def allocate(self) -> Tuple[int, dict]:
        """分配一个 episode slot。返回 (slot_id, slices_dict)。"""
        slot = self._write_idx % self.max_episodes
        self._write_idx += 1
        self._count = min(self._count + 1, self.max_episodes)

        slices = {
            "obs": self.obs_buffer[slot],
            "actions": self.action_buffer[slot],
            "rewards": self.reward_buffer[slot],
            "dones": self.done_buffer[slot],
            "masks": self.mask_buffer[slot],
        }
        return slot, slices

    def finalize_episode(self, slot: int, episode_len: int, policy_version: int):
        """完成一个 episode 的写入。"""
        self.version_buffer[slot] = policy_version

    def read_slot(self, slot: int) -> dict:
        return {
            "obs": self.obs_buffer[slot],
            "actions": self.action_buffer[slot],
            "rewards": self.reward_buffer[slot],
            "dones": self.done_buffer[slot],
            "masks": self.mask_buffer[slot],
            "version": self.version_buffer[slot],
        }
