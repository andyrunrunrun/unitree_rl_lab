from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """步态相位观测（sin/cos 编码）。

    目的：
    - 为策略提供一个“时间相位”信号，使其更容易学习周期性动作（例如交替迈步）。
    - 使用 \(\sin,\cos\) 编码可以避免相位在 0/1 处的跳变问题（连续且可微）。

    输入：
    - **env**：环境对象，要求包含 `num_envs`、`device`、`step_dt` 等属性。
    - **period**：步态周期（秒）。相位会按该周期循环。

    输出：
    - 形状为 `(num_envs, 2)` 的张量：`[sin(2πphase), cos(2πphase)]`。

    实现细节：
    - 该函数依赖 `env.episode_length_buf`（每个并行环境已执行的 step 数）。
      如果环境对象上没有这个字段，会在首次调用时创建并缓存一个全 0 的张量。
    - 这里假设外部训练循环会更新 `env.episode_length_buf`；若没有更新，相位将一直为 0。
    """
    if not hasattr(env, "episode_length_buf"):
        # 每个环境的步数计数器（long），用于相位计算；由上层环境逻辑持续递增
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    # 将离散 step 计数映射到连续时间（秒），再映射到 [0, 1) 的归一化相位
    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    # 2 维相位编码：sin/cos
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase
