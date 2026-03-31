from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""Locomotion 任务奖励/惩罚项集合。

该文件包含一组“可组合”的 reward/penalty 函数，供 `RewardTermCfg` 在环境配置里引用。
每个函数都遵循统一约定：
- 输入第一个参数为 `env: ManagerBasedRLEnv`；
- 返回形状为 `(num_envs,)` 的 1D 张量（每个并行环境一个标量）。

约定与注意事项：
- 这里的很多函数对坐标系有隐含假设：
  - `*_w` 通常表示 world frame（世界坐标系）；
  - `*_b` 通常表示 body frame（机体/根坐标系），具体含义以 IsaacLab 数据字段为准。
- 部分函数会在 `env` 上缓存中间结果（例如 `joint_mirror_joints_cache`），以减少每 step 的查找开销。
"""

# -----------------------------------------------------------------------------
# Joint / actuator 相关惩罚
# -----------------------------------------------------------------------------


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节能耗惩罚（近似机械功率积分）。

    计算方式（逐关节）：
    - 使用 `|qvel| * |torque|` 作为功率的粗略代理，并对关节维度求和。

    直觉：
    - 惩罚高速度与高力矩同时出现的情况，鼓励更省力、更平滑的动作。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """“站稳”惩罚：当指令接近 0 时，惩罚关节偏离默认姿态。

    用途：
    - 在期望站立（速度指令很小）时，避免策略仍做出大幅摆动。

    说明：
    - 返回值是“偏离量”，通常在配置里作为 *负权重* 惩罚项使用。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)

# -----------------------------------------------------------------------------
# 机体姿态 / 姿势相关
# -----------------------------------------------------------------------------


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """朝向奖励：将机体重力投影对齐到目标重力方向（平方核）。

    解释：
    - `asset.data.projected_gravity_b` 通常是“世界重力方向在 body frame 下的表示”（单位向量或近似）。
    - 与 `desired_gravity` 做点乘得到余弦相似度 `cos_dist`（范围约 [-1, 1]）。
    - 线性映射到 [0, 1] 后再平方，使得越接近对齐奖励越高，且对偏差更敏感。

    返回：
    - `(num_envs,)`，值域大致在 [0, 1]。
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """“保持竖直”惩罚：惩罚机体 z 轴不够朝上。

    这里用 `1 - projected_gravity_b[:, 2]` 来衡量与“竖直朝上”的偏差，
    再取平方作为惩罚强度。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """关节姿态偏离惩罚（带“站立时更强”逻辑）。

    计算：
    - `reward = ||q - q_default||`（对关节向量做 L2 范数）
    - 若有运动意图（command > 0）或机器人确实在动（body_vel > threshold），按原值返回；
      否则在站立场景下将惩罚乘以 `stand_still_scale`（通常 > 1），更强地把姿态拉回默认。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)

# -----------------------------------------------------------------------------
# 足端相关 reward/penalty
# -----------------------------------------------------------------------------


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """绊脚/撞障惩罚：足端与“竖直表面”碰撞的启发式检测。

    思路：
    - 若接触力的水平分量远大于竖直分量（forces_xy > 4 * forces_z），
      往往意味着脚踢到竖直障碍面或发生异常侧向撞击。
    - 返回 0/1（float）指示是否发生该现象。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """足端“相对机体”抬脚高度惩罚/奖励（当前实现更像惩罚项）。

    关键点：
    - 将足端位置/速度从 world frame 转到 root(body) frame，再对 z 高度接近 `target_height` 的误差做平方；
    - 用足端水平速度的 `tanh` 作为权重：摆动越快越强调该项；
    - 仅当 command 足够大时启用（避免站立时瞎抬脚）；
    - 额外乘以一个基于机体姿态的缩放（`projected_gravity_b[:, 2]`），在姿态不佳时减小该项影响。

    说明：
    - 该函数输出的是误差累积（越小越好），通常配合 *负权重* 当作惩罚使用。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    # 将每只脚的 (pos, vel) 从 world 旋转到 body(root) frame
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """足端离地间隙奖励（指数核）。

    计算：
    - 误差：`(foot_z - target_height)^2`
    - 权重：足端水平速度的 `tanh(tanh_mult * ||v_xy||)`，摆动越快越强调 clearance
    - 将所有足端的误差加权求和后，做 `exp(-sum / std)` 得到 [0, 1] 区间的奖励

    参数含义：
    - **target_height**：希望足端在摆动相位的典型离地高度（米）
    - **std**：控制指数衰减速度，越小越“挑剔”
    - **tanh_mult**：速度权重的放大因子，越大越快饱和
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """双足间距过近惩罚。

    - 计算两只脚（`asset_cfg.body_ids` 默认应为左右脚）的欧氏距离；
    - 若距离小于 `threshold`，返回正的惩罚量 `threshold - distance`，否则为 0。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """当速度指令为 0 时的“足端接触奖励”。

    直觉：
    - 当期望站立（command 很小）时，鼓励脚保持接触地面，减少抬脚/跳跃等不稳定动作。
    - 这里用“当前接触时间 > 0”判定是否接触，并对足端数量求和。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """腾空/接触时间方差惩罚：约束左右脚节律一致性。

    - 读取接触传感器记录的 `last_air_time` 与 `last_contact_time`；
    - 对两只脚在这些时间上的方差求和作为惩罚；
    - 对时间做 clip（max=0.5）以防异常值主导训练信号。

    注意：
    - 需要 `ContactSensorCfg(track_air_time=True)`，否则会直接抛错。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )

# -----------------------------------------------------------------------------
# 步态（gait）一致性奖励
# -----------------------------------------------------------------------------


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    """步态相位-接触匹配奖励。

    思路：
    - 使用 episode 内的时间相位（按 `period` 周期归一化）生成每条腿的期望相位；
    - 当腿相位 < `threshold` 视为“支撑相（stance）”，否则为“摆动相（swing）”；
    - 将期望的支撑/摆动与真实接触（`is_contact`）做一致性判断，匹配则累加奖励。

    参数：
    - **period**：步态周期（秒）
    - **offset**：每条腿的相位偏置（例如双足反相可用 [0.0, 0.5]）
    - **threshold**：支撑相占比（0~1），越大表示支撑期越长
    - **command_name**：若提供，则只有在 command 足够大时才启用该奖励（避免站立时硬套步态）
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    # (num_envs, 1)：全局相位
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    # (num_envs, num_legs)：每条腿的相位
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        # XNOR：期望接触(stance) 与 实际接触 is_contact 一致则为 True
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward

# -----------------------------------------------------------------------------
# 其它辅助 reward/penalty
# -----------------------------------------------------------------------------


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    """左右对称关节一致性惩罚（镜像约束）。

    用途：
    - 当希望左右肢体动作更对称时，可对成对关节的角度差做平方惩罚。

    参数：
    - **mirror_joints**：形如 `[[left_joint_regex, right_joint_regex], ...]` 的列表。
      每一对会通过 `asset.find_joints` 解析为 joint id，并缓存到 env 上避免重复查找。

    返回：
    - 对所有关节对的平方差求和，并按对数做平均（若为空则返回 0）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
