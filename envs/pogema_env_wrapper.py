"""
Pogema 环境统一封装。
支持四类地图族：random / warehouse / maze / room。

统一 reset/step 接口，暴露每体局部观测（仅局部信息进 policy）。
全局地图/距离场等特权信号仅通过独立方法暴露，供训练期 reward/LaCAM 使用，
**不进 policy 观测**（INVARIANT 2, 3）。
"""
from __future__ import annotations

import os
import numpy as np
import yaml
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy

from pogema import GridConfig, pogema_v0


# ---------------------------------------------------------------------------
# 地图生成
# ---------------------------------------------------------------------------

def _generate_random_map(cfg: dict, seed: Optional[int] = None) -> str:
    """Random 地图由 Pogema 原生 generator 生成，不需要预生成 map string。"""
    return None


def _generate_warehouse_map(cfg: dict, seed: Optional[int] = None) -> str:
    """
    使用 pogema-toolbox 生成 warehouse 地图字符串。
    注意：warehouse generator 使用 '!' 表示空闲、'#' 表示障碍。
    Pogema 期望 '.' 表示空闲、'#' 表示障碍，需要转换。
    """
    from pogema_toolbox.generators.warehouse_generator import (
        WarehouseConfig,
        generate_warehouse,
    )
    wh_cfg = cfg.get("warehouse", {})
    wc = WarehouseConfig(
        wall_width=wh_cfg.get("wall_width", 5),
        wall_height=wh_cfg.get("wall_height", 2),
        walls_in_row=wh_cfg.get("walls_in_row", 5),
        walls_rows=wh_cfg.get("walls_rows", 5),
        bottom_gap=wh_cfg.get("bottom_gap", 5),
        horizontal_gap=wh_cfg.get("horizontal_gap", 1),
        vertical_gap=wh_cfg.get("vertical_gap", 3),
    )
    map_str = generate_warehouse(wc)
    # warehouse generator 用 '!' 表示空闲，转换为 Pogema 期望的 '.'
    map_str = map_str.replace('!', '.')
    return map_str


def _generate_maze_map(cfg: dict, seed: Optional[int] = None) -> str:
    """使用 pogema-toolbox MazeGenerator 生成 maze 地图字符串。"""
    from pogema_toolbox.generators.maze_generator import MazeGenerator

    maze_cfg = cfg.get("maze", {})
    rng = np.random.default_rng(seed)

    width = int(rng.integers(
        maze_cfg.get("width_min", 20), maze_cfg.get("width_max", 64) + 1
    ))
    height = int(rng.integers(
        maze_cfg.get("height_min", 20), maze_cfg.get("height_max", 64) + 1
    ))
    max_size = cfg.get("map_size", 64)
    width = min(width, max_size)
    height = min(height, max_size)

    obstacle_density = float(rng.uniform(
        maze_cfg.get("obstacle_density_min", 0.2),
        maze_cfg.get("obstacle_density_max", 0.4),
    ))
    wall_components = int(rng.integers(
        maze_cfg.get("wall_components_min", 1),
        maze_cfg.get("wall_components_max", 8) + 1,
    ))
    go_straight = float(rng.uniform(0.75, 0.85))

    return MazeGenerator.generate_maze(
        width=width,
        height=height,
        obstacle_density=obstacle_density,
        wall_components=wall_components,
        go_straight=go_straight,
        seed=seed,
    )


def _generate_room_map(cfg: dict, seed: Optional[int] = None) -> str:
    """使用 MazeGenerator 生成类 room 地图（更多开阖区域，模拟房间+门洞）。"""
    from pogema_toolbox.generators.maze_generator import MazeGenerator

    room_cfg = cfg.get("room", {})
    rng = np.random.default_rng(seed)

    width = int(rng.integers(
        room_cfg.get("width_min", 20), room_cfg.get("width_max", 64) + 1
    ))
    height = int(rng.integers(
        room_cfg.get("height_min", 20), room_cfg.get("height_max", 64) + 1
    ))
    max_size = cfg.get("map_size", 64)
    width = min(width, max_size)
    height = min(height, max_size)

    obstacle_density = float(rng.uniform(
        room_cfg.get("obstacle_density_min", 0.1),
        room_cfg.get("obstacle_density_max", 0.3),
    ))
    wall_components = int(rng.integers(
        room_cfg.get("wall_components_min", 2),
        room_cfg.get("wall_components_max", 12) + 1,
    ))
    go_straight = float(rng.uniform(0.65, 0.80))

    return MazeGenerator.generate_maze(
        width=width, height=height,
        obstacle_density=obstacle_density,
        wall_components=wall_components,
        go_straight=go_straight,
        seed=seed,
    )


_MAP_GENERATORS = {
    "random": _generate_random_map,
    "warehouse": _generate_warehouse_map,
    "maze": _generate_maze_map,
    "room": _generate_room_map,
}


# ---------------------------------------------------------------------------
# 主封装类
# ---------------------------------------------------------------------------

class PogemaWrapper:
    """
    Pogema 环境统一封装。

    用法:
        wrapper = PogemaWrapper("configs/env_maze.yaml")
        obs_list, info = wrapper.reset()
        for _ in range(max_steps):
            actions = [...]
            obs_list, rewards, dones, info = wrapper.step(actions)

    **策略只通过 obs_list 获取局部观测**（INVARIANT 2）。
    全局信息（grid / starts / goals）通过独立属性访问，仅供训练期 reward/LaCAM。
    """

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self._raw_cfg = yaml.safe_load(f)

        self.map_type: str = self._raw_cfg["map_type"]
        self.map_size: int = self._raw_cfg.get("map_size", 64)
        self.num_agents: int = self._raw_cfg.get("num_agents", 8)
        self.obs_radius: int = self._raw_cfg.get("obs_radius", 5)
        self.max_episode_steps: int = self._raw_cfg.get("max_episode_steps", 256)
        self.collision_system: str = self._raw_cfg.get("collision_system", "soft")
        self.on_target: str = self._raw_cfg.get("on_target", "restart")

        self._env = None
        self._episode_step: int = 0
        self._seed: Optional[int] = None

        # 当前 episode 的地图全局信息（仅供训练期 reward/LaCAM 使用，不进 policy）
        self._global_grid: Optional[np.ndarray] = None
        self._starts: Optional[List[Tuple[int, int]]] = None
        self._goals: Optional[List[Tuple[int, int]]] = None

    # ---- 属性（特权信号，仅供训练期基础设施使用）----------------------

    @property
    def global_grid(self) -> Optional[np.ndarray]:
        """全局障碍物栅格 (H, W), 0=free, 1=obstacle。仅供 reward/LaCAM，不进 policy。"""
        return self._global_grid

    @property
    def starts(self) -> Optional[List[Tuple[int, int]]]:
        """各智能体的起始位置列表。"""
        return self._starts

    @property
    def goals(self) -> Optional[List[Tuple[int, int]]]:
        """各智能体的目标位置列表。"""
        return self._goals

    @property
    def unwrapped_env(self):
        """底层 Pogema 环境实例（用于访问 Pogema 原生 API）。"""
        return self._env

    # ---- 核心 API ----------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[List[np.ndarray], dict]:
        """
        重置环境，生成新地图。

        Returns:
            obs_list: list of np.ndarray, 每体原生观测 shape=(3, K, K)
            info: dict with metadata
        """
        self._seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        self._episode_step = 0
        self._env = self._create_env()

        result = self._env.reset(seed=self._seed)
        # Pogema 2.x (Gymnasium 风格): reset 返回 (obs_list, info_list)
        # 其中 info_list 是 list[dict]，每体一个 info dict
        if isinstance(result, tuple) and len(result) == 2:
            obs_list, env_info = result
        else:
            obs_list, env_info = result, [{}] * self.num_agents

        # env_info 是 per-agent list[dict]，聚合为统一 info
        if isinstance(env_info, list):
            # 从 per-agent info 提取公共字段
            agg_info = {
                "num_active": sum(1 for d in env_info if d.get("is_active", True)),
            }
        elif isinstance(env_info, dict):
            agg_info = dict(env_info)
        else:
            agg_info = {}

        # 提取全局信息
        self._extract_global_info()

        info = {
            "map_type": self.map_type,
            "map_size": self.map_size,
            "num_agents": self.num_agents,
            "episode_step": self._episode_step,
            **agg_info,
        }
        return list(obs_list), info

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], dict]:
        """
        执行一步。

        Args:
            actions: 每个智能体的动作 id ∈ {0,1,2,3,4}。

        Returns:
            obs_list: 每体原生观测
            rewards: 每体环境原始奖励（不含 shaping）
            dones: 每体 done 标志
            info: 附加信息
        """
        self._episode_step += 1
        result = self._env.step(actions)

        if isinstance(result, tuple) and len(result) >= 4:
            obs_list = list(result[0])
            rewards = list(result[1])
            dones = list(result[2])
            # 兼容 Gymnasium 5-元组格式 (obs, reward, terminated, truncated, info)
            if len(result) == 5:
                terminated = result[2]
                truncated = result[3]
                dones = [bool(t) or bool(tr) for t, tr in zip(terminated, truncated)]
                infos = result[4]
            else:
                infos = result[3] if len(result) > 3 else {}
        else:
            obs_list = list(result)
            rewards = [0.0] * self.num_agents
            dones = [False] * self.num_agents
            infos = {}

        return obs_list, rewards, dones, infos

    def get_targets_xy(self) -> List[Tuple[int, int]]:
        """
        获取各智能体当前目标坐标。
        INVARIANT 6: 目标相对方向是合法的局部特征（智能体始终知道自己的 goal）。
        """
        if self._env is None:
            return [(0, 0)] * self.num_agents
        try:
            targets = self._env.get_targets_xy()
            return [tuple(t) for t in targets]
        except Exception:
            return [(0, 0)] * self.num_agents

    def get_agents_xy(self) -> List[Tuple[int, int]]:
        """获取各智能体当前位置（仅供训练期诊断/LaCAM）。"""
        if self._env is None:
            return [(0, 0)] * self.num_agents
        try:
            agents = self._env.get_agents_xy()
            return [tuple(a) for a in agents]
        except Exception:
            return [(0, 0)] * self.num_agents

    # ---- 内部 --------------------------------------------------------

    def _create_env(self):
        """根据 map_type 创建对应的 Pogema 环境。"""
        generator = _MAP_GENERATORS.get(self.map_type)
        if generator is None:
            raise ValueError(f"Unknown map_type: {self.map_type}")

        map_str = generator(self._raw_cfg, seed=self._seed)

        gc_kwargs = dict(
            size=self.map_size,
            num_agents=self.num_agents,
            max_episode_steps=self.max_episode_steps,
            obs_radius=self.obs_radius,
            collision_system=self.collision_system,
            on_target=self.on_target,
            seed=self._seed,
        )
        # 注意：不设 observation_type！
        # Pogema 2.x 默认返回标准 (3,K,K) numpy array。
        # 显式设 'POMAPF' 会改为 dict 格式，不符合设计文档的观测通道约定。

        # ★ map 必须在构造函数传入（Pydantic validator 将 string→list[list[int]]），
        #    后设属性 gc.map = str 会跳过 validator，导致 grid.py 解析失败。
        if map_str is not None:
            gc_kwargs["map"] = map_str
            gc_kwargs["map_name"] = f"{self.map_type}-seed-{self._seed}"

        gc = GridConfig(**gc_kwargs)
        return pogema_v0(grid_config=gc)

    def _extract_global_info(self):
        """从环境提取全局栅格和起终点（仅供训练期基础设施）。"""
        if self._env is None:
            return
        try:
            self._global_grid = self._env.get_global_obstacles().astype(np.int32)
        except Exception:
            try:
                # Fallback: 直接访问 grid
                obstacles = self._env.grid.get_obstacles()
                self._global_grid = np.asarray(obstacles, dtype=np.int32)
            except Exception:
                self._global_grid = None
        try:
            self._starts = self.get_agents_xy()
            self._goals = self.get_targets_xy()
        except Exception:
            self._starts = None
            self._goals = None


# ---------------------------------------------------------------------------
# 便捷工厂函数
# ---------------------------------------------------------------------------

def make_env(config_path: str) -> PogemaWrapper:
    """从 YAML 配置文件创建 PogemaWrapper。"""
    return PogemaWrapper(config_path)


def make_env_from_dict(cfg: dict) -> PogemaWrapper:
    """从配置字典创建（用于训练脚本中动态覆盖参数）。"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp_path = f.name
    wrapper = PogemaWrapper(tmp_path)
    os.unlink(tmp_path)
    return wrapper
