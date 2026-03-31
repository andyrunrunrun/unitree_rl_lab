from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""Mimic（动作模仿）任务的奖励函数集合。

核心思想：把“参考动作（motion）”看作一个随时间变化的目标轨迹，然后对比：

- 参考动作的 anchor（通常是 torso_link）的 **位置/朝向**
- 参考动作中若干关键 body（脚踝/手腕/躯干等）的 **相对位置/相对姿态**
- 以及这些 body 的 **线速度/角速度**

并用指数核（exponential kernel）把误差映射到 \((0, 1]\) 的奖励：

\n    reward = exp(- error / std^2)\n

这样做的好处是：误差接近 0 时梯度更稳定，误差很大时奖励自动饱和到接近 0（不会出现极端的大负值）。

在 env_cfg 里，你会看到这些函数通过 `RewTerm(func=..., weight=..., params={...})` 的方式被组合成总奖励。
"""


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    """把 body_names（子集）映射为 MotionCommand.cfg.body_names 的索引列表。

    - `command.cfg.body_names` 是 Mimic 任务里“被跟踪的关键刚体集合”（例如 pelvis、脚踝、手腕等）。
    - 某些 reward/termination 只关心其中的末端（例如脚踝/手腕），这时用 body_names 传入子集。
    """
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """奖励 anchor 的全局位置跟踪精度（参考 vs 机器人）。

    - anchor 一般选择 `torso_link`，代表“身体主干”。
    - 误差以 L2^2 计算（对 xyz 求平方和），再走 exp 核映射到 (0,1]。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """奖励 anchor 的全局姿态跟踪精度（参考 vs 机器人）。"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """奖励关键 body 的相对位置（以 anchor 对齐后的坐标系）跟踪精度。

    这里用的是 `MotionCommand.body_pos_relative_w`，它把参考动作“对齐”到机器人当前 anchor（主要对齐 yaw 和位置），
这样 reward 更关注姿态/相对结构，而不是绝对世界坐标的漂移。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """奖励关键 body 的相对姿态跟踪精度（参考 vs 机器人）。"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """奖励关键 body 的全局线速度跟踪精度。"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """奖励关键 body 的全局角速度跟踪精度。"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """一个与接触时序相关的奖励/指标：鼓励触地时间分布满足阈值。

    这个函数在 Mimic 中通常用于辅助约束脚部接触模式（比如避免长时间悬空或拖地）。
    返回值是按脚汇总的分数（越大代表越符合阈值条件）。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
