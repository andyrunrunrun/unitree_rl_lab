import gymnasium as gym

"""注册 G1 29DOF 的 Mimic 任务（Dance-102）。

Gymnasium/Isaac Lab 的环境注册表依赖 `gym.register(...)`。
训练脚本 `scripts/rsl_rl/train.py` 会先导入 `unitree_rl_lab.tasks`（触发这些注册代码），
然后你就可以在命令行里通过 `--task Unitree-G1-29dof-Mimic-Dance-102` 选择该环境。

三个 entry point 的含义：
- **env_cfg_entry_point**：训练用环境配置（RobotEnvCfg）
- **play_env_cfg_entry_point**：推理/可视化用环境配置（RobotPlayEnvCfg，通常更轻量）
- **rsl_rl_cfg_entry_point**：PPO 训练超参数（BasePPORunnerCfg）
"""

gym.register(
    id="Unitree-G1-29dof-Mimic-Dance-102",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
