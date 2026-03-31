# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Mimic 任务使用的 RSL-RL PPO 训练超参数。

    这份配置与 locomotion（速度控制）非常相似，但有几个关键差异：

    - **max_iterations=30000**：Mimic 通常更“像监督学习 + RL 微调”，收敛速度更快一些；
    - **save_interval=500**：保存间隔更大，减少磁盘占用；
    - **entropy_coef=0.005**：探索熵更小，鼓励更稳定地跟踪参考动作（减少随机性）。

    读代码时建议你先抓住三个核心量：

    - **num_steps_per_env**：每次 rollout 每个环境采样多少步
    - **policy 网络结构**：Actor/Critic 的 MLP 隐层维度
    - **algorithm**：PPO 的 clip、学习率、mini-batch、GAE 等
    """

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
