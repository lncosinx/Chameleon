"""地图采样调度器：课程式采样 + 自适应失败率加权。"""
import numpy as np
from typing import Dict, List, Optional


class MapScheduler:
    """
    地图类型采样调度器。

    前期: random=0.45, warehouse=0.20, maze=0.20, room=0.15
    中期: random=0.30, warehouse=0.30, maze=0.25, room=0.15
    后期: random=0.20, warehouse=0.30, maze=0.30, room=0.20
    """

    def __init__(self, schedule: Dict[str, Dict[str, float]] = None):
        if schedule is None:
            schedule = {
                "early": {"random": 0.45, "warehouse": 0.20, "maze": 0.20, "room": 0.15},
                "mid": {"random": 0.30, "warehouse": 0.30, "maze": 0.25, "room": 0.15},
                "late": {"random": 0.20, "warehouse": 0.30, "maze": 0.30, "room": 0.20},
            }
        self.schedule = schedule
        self._rng = np.random.default_rng()
        self._fail_rates: Dict[str, float] = {}

    def get_probs(self, iter_num: int, total_iters: int,
                  adaptive: bool = False) -> Dict[str, float]:
        """获取当前迭代的地图采样概率。"""
        progress = iter_num / max(total_iters, 1)
        if progress < 0.33:
            phase = "early"
        elif progress < 0.66:
            phase = "mid"
        else:
            phase = "late"

        probs = dict(self.schedule[phase])

        if adaptive and self._fail_rates:
            eta = 0.5
            for m in probs:
                if m in self._fail_rates:
                    probs[m] *= np.exp(eta * self._fail_rates[m])
            # renormalize
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}

        return probs

    def sample(self, iter_num: int, total_iters: int,
               adaptive: bool = False) -> str:
        """采样一个地图类型。"""
        probs = self.get_probs(iter_num, total_iters, adaptive)
        types = list(probs.keys())
        weights = [probs[t] for t in types]
        return self._rng.choice(types, p=weights)

    def update_fail_rate(self, map_type: str, fail_rate: float):
        self._fail_rates[map_type] = fail_rate


def expert_ratio(cur_iter: int, total_iters: int,
                 start: float = 0.9, end: float = 0.5,
                 decay_iters: int = 8000, bc_offset: int = 0) -> float:
    """课程式 expert/self-play 混合比例（线性衰减）。

    bc_offset: 两阶段 BC 预训练步数。BC 期（cur_iter < bc_offset）self-play rollout 被
    跳过、sp_buf 为空，此时 er 必须 = 1.0：否则 n_expert=int(B·er)<B，剩下的 n_selfplay
    个 batch 槽既无 self-play 数据填充、也不补给专家，等于白扔 (1-er) 比例的 batch
    （er=0.9 时 BC 实际只在 28/32 序列上训）。故 BC 期返回 1.0（全 batch 给专家）。
    iter=bc_offset 时 er 自然跌到 start（self-play 从 1-start 温和起步），随后在
    [bc_offset, bc_offset+decay_iters] 上线性衰减到 end。
    """
    if cur_iter < bc_offset:
        return 1.0
    eff_iter = cur_iter - bc_offset
    if eff_iter >= decay_iters:
        return end
    return start - (start - end) * eff_iter / decay_iters
