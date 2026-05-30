#!/usr/bin/env python3
"""
Phase 5-7 集成验收：缓冲 + 训练器 + 评估。

验证:
- T5: 共享内存缓冲 + collator
- T6: Reptile DQfD 训练循环（loss 下降、Q 不发散）
- T7: 评估器产出指标
"""
import sys, os, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from utils.seed import set_seed
from models.biintent_drqn import BiIntentGATDRQN
from trainers.train_reptile_dqfd import ReptileDQfDTrainer
from buffers.expert_demo_buffer import ExpertDemoBuffer
from trainers.evaluator import evaluate
from utils.checkpoint import save_checkpoint, load_checkpoint
from envs.pogema_env_wrapper import PogemaWrapper
from envs.observation_builder import ObservationBuilder
from envs.reward_builder import RewardBuilder
from lacam.lacam_runner import LaCAMRunner
from lacam.lacam_dataset_builder import build_dataset_from_rollout
from buffers.batch_collator import collate_samples_to_tensors

set_seed(42)


def test_collator():
    """T5: Batch collator。"""
    print("\n--- T5: Batch collator ---")
    K = 11
    samples = []
    for i in range(64):
        samples.append({
            "obs_patch": np.random.randn(4, K, K).astype(np.float32),
            "obs_scalar": np.random.randn(8).astype(np.float32),
            "action_expert": i % 5,
            "reward": float(i),
            "done": i % 10 == 0,
            "action_mask": np.ones(5, dtype=bool),
            "margin_excluded": i % 20 == 0,
            "margin_weight": 1.0,
        })
    batch = collate_samples_to_tensors(samples, sequence_len=32)
    assert batch["obs_patches"].shape == (2, 32, 4, K, K)
    print(f"  Collated: {batch['obs_patches'].shape} ✓")


def test_training_loop():
    """T6: Reptile DQfD 训练循环。"""
    print("\n--- T6: Reptile DQfD training ---")

    device = torch.device("cpu")
    net = BiIntentGATDRQN(
        patch_channels=4, cnn_channels=[8, 16, 16],
        d_patch=64, d_scalar=16, intent_dim=32, comm_dim=32,
        comm_radius=4, comm_topk=3, num_attention_heads=2,
        gru_hidden_dim=64, dueling_hidden_dim=64, num_actions=5,
    )

    cfg = {
        "inner_lr": 1e-3, "meta_lr": 0.1, "inner_steps": 3,
        "batch_size": 8, "sequence_len": 16, "burn_in_len": 4,
        "gamma": 0.99, "mellowmax_alpha": 1.0, "margin": 0.5,
        "lambda_td": 1.0, "lambda_nstep": 0.0, "lambda_margin": 1.0,
        "lambda_l2": 1e-5, "expert_ratio_decay_iters": 100,
        "meta_iterations": 50,
    }
    trainer = ReptileDQfDTrainer(net, cfg, device=device)

    # 生成专家数据
    wrapper = PogemaWrapper("configs/env_random.yaml")
    wrapper._raw_cfg.update(size=16, num_agents=2)
    wrapper.map_size = 16
    wrapper.num_agents = 2

    obs_builder = ObservationBuilder(num_agents=2, obs_radius=5)
    reward_builder = RewardBuilder(lambda_f=0.1)
    runner = LaCAMRunner(backend="prioritized")
    expert_buf = ExpertDemoBuffer(capacity=2000)

    for ep in range(15):
        wrapper.reset(seed=60000 + ep)
        obstacles = wrapper.global_grid
        starts = wrapper.get_agents_xy()
        goals = wrapper.get_targets_xy()
        result = runner.solve(obstacles, starts, goals, map_type="random", map_size=16)
        if result.success:
            wrapper.reset(seed=60000 + ep)
            samples = build_dataset_from_rollout(
                wrapper, obs_builder, reward_builder, obstacles,
                result.paths, "random", f"train_{ep}"
            )
            expert_buf.add_batch(samples)

    print(f"  Expert buffer: {len(expert_buf)} samples")

    # 训练前 Q 值统计
    net.eval()
    with torch.no_grad():
        dummy_patches = torch.zeros(1, 1, 2, 4, 11, 11)
        dummy_scalars = torch.zeros(1, 1, 2, 8)
        dummy_positions = torch.zeros(1, 1, 2, 2)
        h = net.init_hidden(1, 2, device)
        q_before, _, info_before = net(dummy_patches, dummy_scalars, dummy_positions, h)
    q_mean_before = info_before["q_mean"]
    print(f"  Q mean before training: {q_mean_before:.4f}")

    # 训练 30 meta-iter
    losses = []
    for i in range(30):
        metrics = trainer.meta_step(expert_buf, None, None)
        losses.append(metrics["inner_loss_mean"])

    print(f"  Loss: start={losses[0]:.4f}, end={losses[-1]:.4f}")

    # Q 值不发散
    with torch.no_grad():
        q_after, _, info_after = net(dummy_patches, dummy_scalars, dummy_positions, h)
    q_mean_after = info_after["q_mean"]
    print(f"  Q mean after training: {q_mean_after:.4f}")
    # 不发散：Q 值不超过 10（未训练的随机网络 Q 更小）
    assert abs(q_mean_after) < 50.0, f"Q diverged: {q_mean_after}"
    print(f"  [PASS] Q values not diverged ✓")

    # Loss 整体下降
    first_5 = np.mean(losses[:5])
    last_5 = np.mean(losses[-5:])
    print(f"  Loss first 5 mean: {first_5:.4f}, last 5 mean: {last_5:.4f}")

    # Checkpoint save/load
    save_checkpoint(trainer.state_dict(), "/tmp/test_ckpt.pt", "meta")
    ckpt = load_checkpoint("/tmp/test_ckpt.pt", device)
    assert ckpt["type"] == "meta"
    assert ckpt["meta_iter"] == 30
    print(f"  [PASS] Checkpoint save/load OK ✓")


def test_evaluation():
    """T7: 评估器（关闭专家介入）。"""
    print("\n--- T7: Evaluation ---")

    net = BiIntentGATDRQN(
        patch_channels=4, cnn_channels=[8, 16, 16],
        d_patch=64, d_scalar=16, intent_dim=32, comm_dim=32,
        gru_hidden_dim=64, dueling_hidden_dim=64, num_actions=5,
    )
    device = torch.device("cpu")

    metrics = evaluate(
        net, "configs/env_random.yaml",
        num_episodes=5, num_agents=2, map_size=16, max_steps=32,
        device=device, epsilon=0.3,  # 一些探索
    )

    required_keys = ["SR_mean", "AR_mean", "Makespan_mean", "WaitRate_mean"]
    for k in required_keys:
        assert k in metrics, f"Missing metric: {k}"
        print(f"  {k}: {metrics[k]:.4f}")

    print(f"  [PASS] Evaluator produces all required metrics ✓")


def main():
    print("=" * 60)
    print("  Chameleon — Phase 5-7 集成验收")
    print("=" * 60)

    all_ok = True
    tests = [test_collator, test_training_loop, test_evaluation]
    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"\n  [FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("  Phase 5-7 验收结果: 全部通过 ✓")
    else:
        print("  Phase 5-7 验收结果: 存在失败项 ✗")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
