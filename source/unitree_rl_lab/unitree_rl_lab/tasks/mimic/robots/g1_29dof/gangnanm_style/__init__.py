import gymnasium as gym

"""注册 G1 29DOF 的 Mimic 任务（Gangnam Style）。

与 `dance_102` 的注册逻辑相同，区别仅在于任务 id 和对应的 `tracking_env_cfg.py` 中使用的动作数据文件不同。

提示：你可以用 `./unitree_rl_lab.sh -l` 查看所有已注册的 Unitree 任务。
"""

gym.register(
    id="Unitree-G1-29dof-Mimic-Gangnanm-Style",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
