import datetime

from common import *

# 快速 smoke / 单策略配置（不做 22 路 ensemble），用于先确认 EPH 能跑通、
# 并给出"纯学习策略"(pure comms policy) 的数字，作为对我们纯通信网络的公平对照。
use_wandb = False
project = "FinalTest"
name = "eph_quick"
run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

model_target = "src.models.eph.Network"
save_path = "./saved_models/eph_net"

use_wandb_test = False
test_folder = "./test_set"
data_mode = "normal"  # 读取 {len}length_{n}agents_{d}density.pth 格式的测试集

density = 0.3
# 只跑 40-length 少量 agent 档，快速验证
test_env_settings = (
    (40, 8, density),
    (40, 16, density),
    (40, 32, density),
    (40, 64, density),
)

# 单策略：use_stepinfer_method=False → 纯网络前向（无 A* 引导、无 AEP）
use_stepinfer_method = False
astar_type = None
active_agent_radius = None
use_aep = False
aep_distance_threshold = None
aep_use_q_value_only = False
aep_astar_type = None

# 关闭 ensemble
ensemble = None
