from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """线速度指令课程学习（动态扩大采样范围）。

    该函数会在训练过程中根据最近一个 episode 的“速度跟踪奖励”表现，逐步放宽 `base_velocity`
    指令的采样范围（`command_term.cfg.ranges`），但始终不超过预设的 `limit_ranges`。

    触发条件（按实现）：
    - 仅在 episode 边界（`common_step_counter % max_episode_length == 0`）检查；
    - 若最近 episode 的平均奖励（记作 r_bar）超过该奖励项权重的 0.8 倍，
      则将 `lin_vel_x/lin_vel_y` 的范围上下界各扩大 0.1。

    参数：
    - **env**：IsaacLab 的 manager-based RL 环境对象。
    - **env_ids**：参与统计的环境 id（并行环境中的子集）。
    - **reward_term_name**：用于评估是否“学会了”的奖励项名称（默认 `track_lin_vel_xy`）。

    返回：
    - 当前 `ranges.lin_vel_x` 的上界（张量标量）。该返回值常用于记录/可视化课程进度。

    说明：
    - 这里直接读取了 `reward_manager._episode_sums`（内部字段），属于“实用但有耦合”的实现；
      若 IsaacLab 上游接口变化，需要同步调整。
    - 该函数会 *就地修改* `command_term.cfg.ranges`，因此会影响之后的 command 采样分布。
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    # 最近一个 episode 的平均奖励（按秒归一化），用于跨不同 episode_length 的可比性
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    # 在 episode 边界更新课程（避免每个 step 都抖动范围）
    if env.common_step_counter % env.max_episode_length == 0:
        # 当跟踪奖励达到阈值：扩大线速度指令的采样范围
        if reward > reward_term.weight * 0.8:
            # [-0.1, +0.1] 同时作用于 (min, max)，使区间向两侧对称扩张
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    # 这里返回“当前上界”作为课程进度信号（比如用于日志曲线）
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    """角速度指令课程学习（动态扩大 yaw 角速度范围）。

    逻辑与 `lin_vel_cmd_levels` 相同，只是将课程对象换成 `ranges.ang_vel_z`，
    并通过 `track_ang_vel_z`（默认）这项奖励来判断是否达到放宽条件。

    返回：
    - 当前 `ranges.ang_vel_z` 的上界（张量标量），通常用于记录课程进度。
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
