from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommand
from unitree_rl_lab.tasks.mimic.mdp.rewards import _get_body_indexes

"""Mimic（动作模仿）任务的终止条件集合。

终止条件的设计目标是：当机器人已经“明显跟丢参考动作”或出现“危险/不可恢复姿态”时，
尽快结束当前 episode，让训练把采样预算用在更有学习价值的片段上。

常见终止类型：
- **anchor 偏差过大**：整体躯干（torso_link）高度/位置/姿态跟丢
- **末端偏差过大**：脚踝/手腕等关键末端的高度或位置偏差过大

这些条件通常在 env_cfg 中作为 `DoneTerm(func=..., params=...)` 配置。
"""


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """anchor 的三维位置误差超过阈值则判失败。"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """anchor 的高度（z）误差超过阈值则判失败。

只看 z 的原因：Mimic 里参考动作常会被做 yaw/位置对齐（减少平移漂移影响），
高度偏差往往更能反映“是否摔倒/蹲塌/跳起过高”这种失稳现象。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """anchor 的姿态偏差超过阈值则判失败。

这里用 projected gravity 的 z 分量差来衡量“是否足够直立/对齐”，
相比直接用四元数差异，它对某些自由度（尤其是 yaw）不那么敏感，更贴合“是否摔倒”的判定。
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """关键 body 的三维位置误差任意一个超过阈值，则判失败。"""
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """关键 body 的高度（z）误差任意一个超过阈值，则判失败。

常用于末端（脚踝、手腕）：末端高度错得太离谱时，往往意味着动作已经崩坏或发生碰撞异常。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)
