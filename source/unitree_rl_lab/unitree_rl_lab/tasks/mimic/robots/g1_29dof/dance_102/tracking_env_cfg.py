from __future__ import annotations  # 启用新的类型注释支持

import os  # 导入操作系统相关的功能（如文件路径操作）

import isaaclab.sim as sim_utils  # 导入isaaclab仿真工具模块并重命名为sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg # 导入关节结构配置和基础资产配置
from isaaclab.envs import ManagerBasedRLEnvCfg  # 导入基于管理器的强化学习环境配置
from isaaclab.managers import EventTermCfg as EventTerm  # 导入事件终止配置并重命名为EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # 导入观测组配置并重命名为ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm  # 导入单项观测配置并重命名为ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm  # 导入奖励项配置并重命名为RewTerm
from isaaclab.managers import SceneEntityCfg  # 导入场景实体配置
from isaaclab.managers import TerminationTermCfg as DoneTerm  # 导入终止项配置并重命名为DoneTerm
from isaaclab.scene import InteractiveSceneCfg  # 导入可交互场景配置
from isaaclab.sensors import ContactSensorCfg  # 导入接触传感器配置
from isaaclab.terrains import TerrainImporterCfg  # 导入地形导入器配置

##
# Pre-defined configs
##
from isaaclab.utils import configclass  # 导入配置类装饰器
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # 导入加性均匀噪声配置并重命名为Unoise

import unitree_rl_lab.tasks.mimic.mdp as mdp  # 导入unitree专用MDP模块并改名为mdp
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_ACTION_SCALE  # 导入G1 29DOF mimic动作缩放系数
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG as ROBOT_CFG  # 导入G1 29DOF mimic机器人资产配置重命名为ROBOT_CFG

"""G1 29DOF Mimic（动作模仿/动作跟踪）环境配置：Dance-102。

这份配置与 `locomotion/velocity_env_cfg.py` 的结构很像（scene/commands/actions/observations/rewards/terminations/events），
但核心差异在于：

- **Commands** 不是速度指令，而是 `MotionCommand`：它会从 NPZ 参考动作里读取当前时间步的关节/身体目标；
- **Rewards** 主要由“跟踪误差”构成：机器人姿态/位置/速度与参考动作的误差越小，奖励越高；
- **Play 配置** 通常会把 `num_envs` 降到 1，且把 episode 设得很长，以便完整播放动作序列。

入门阅读顺序建议：
1) `CommandsCfg.motion`（看 motion_file/anchor/body_names 的含义）
2) `ObservationsCfg.policy`（策略到底看到了什么）
3) `RewardsCfg`（训练信号来自哪些误差）
4) `TerminationsCfg`（哪些偏差会直接判失败）
5) 最后再回到 `mdp/commands.py` 看 `MotionCommand` 的实现细节
"""

##
# Scene definition 场景定义
##

VELOCITY_RANGE = {  # 定义6自由度速度范围
    "x": (-0.5, 0.5),  # x轴线速度范围
    "y": (-0.5, 0.5),  # y轴线速度范围
    "z": (-0.2, 0.2),  # z轴线速度范围
    "roll": (-0.52, 0.52),  # 滚转角速度范围
    "pitch": (-0.52, 0.52),  # 俯仰角速度范围
    "yaw": (-0.78, 0.78),  # 偏航角速度范围
}


@configclass
class RobotSceneCfg(InteractiveSceneCfg):  # 继承交互式场景基类
    """Mimic 场景配置（地面 + G1 + 传感器 + 光照）。"""

    # ground terrain 地面地形
    terrain = TerrainImporterCfg(  # 创建地形导入配置
        prim_path="/World/ground",  # 地形资产路径
        terrain_type="plane",  # 地形为平面
        collision_group=-1,  # 碰撞组默认-1
        physics_material=sim_utils.RigidBodyMaterialCfg(  # 设置物理材质
            friction_combine_mode="multiply",  # 摩擦叠加方式为相乘
            restitution_combine_mode="multiply",  # 回复叠加方式为相乘
            static_friction=1.0,  # 静摩擦系数
            dynamic_friction=1.0,  # 动摩擦系数
        ),
        visual_material=sim_utils.MdlFileCfg(  # 可视化材质配置
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",  # 材质文件路径
            project_uvw=True,  # 使用UV投影
        ),
    )
    # robots 机器人资产
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # 复制ROBOT_CFG并设置路径
    # lights 灯光
    light = AssetBaseCfg(
        prim_path="/World/light",  # 灯光资产路径
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),  # 设置为远距离光源
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",  # 天空灯资产路径
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),  # 设置为圆顶光源
    )
    contact_forces = ContactSensorCfg(  # 接触力传感器配置
        prim_path="{ENV_REGEX_NS}/Robot/.*",  # 监控所有机器人link
        history_length=3,  # 记录3帧历史
        track_air_time=True,  # 追踪空中时间
        force_threshold=10.0,  # 力阈值
        debug_vis=True  # 开启debug可视化
    )


##
# MDP settings MDP配置
##


@configclass
class CommandsCfg:  # 定义命令空间
    """MDP 的命令（Command）定义。

    Mimic 任务里最重要的命令是 `motion`：
    - 读取参考动作（NPZ）
    - 维护每个并行环境的“参考动作时间步”（time_steps）
    - 提供给 observation/reward/termination 使用的目标量（anchor/body/joint 的目标）
    """

    motion = mdp.MotionCommandCfg(  # 定义动作模仿命令配置
        asset_name="robot",  # 目标资产名
        # generate npz file before training
        # python python scripts/mimic/csv_to_npz.py -f path/to/G1_Take_102.bvh_60hz.csv --input_fps 60
        motion_file=f"{os.path.dirname(__file__)}/G1_Take_102.bvh_60hz.npz",  # 参考动作文件路径
        anchor_body_name="torso_link",  # 锚定身体link名
        resampling_time_range=(1.0e9, 1.0e9),  # 重采样时间范围
        debug_vis=True,  # 启用调试可视化
        pose_range={  # 姿态扰动范围
            "x": (-0.05, 0.05),  # x方向扰动
            "y": (-0.05, 0.05),  # y方向扰动
            "z": (-0.01, 0.01),  # z方向扰动
            "roll": (-0.1, 0.1),  # roll扰动
            "pitch": (-0.1, 0.1),  # pitch扰动
            "yaw": (-0.2, 0.2),  # yaw扰动
        },
        velocity_range=VELOCITY_RANGE,  # 速度扰动范围
        joint_position_range=(-0.1, 0.1),  # 关节角扰动
        body_names=[  # 跟踪的身体link名称列表
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
    )


@configclass
class ActionsCfg:  # 定义动作空间配置
    """动作空间定义（策略输出什么）。

    这里使用关节位置动作（position targets）：
    - `use_default_offset=True`：以机器人默认关节角作为 offset
    - `scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE`：每个关节一个缩放系数（由力矩上限/刚度推得）

    这类设计的直觉是：让策略输出的是“相对默认姿态的小偏移”，更稳定、更容易学到可部署的动作。
    """

    JointPositionAction = mdp.JointPositionActionCfg(  # 关节位置动作配置
        asset_name="robot",  # 目标资产名
        joint_names=[".*"],  # 匹配所有关节
        scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,  # 使用关节缩放
        use_default_offset=True  # 动作偏移以默认角度为基准
    )


@configclass
class ObservationsCfg:  # 定义观测空间配置
    """观测定义：策略（policy）与价值网络（critic）分别看什么。"""

    @configclass
    class PolicyCfg(ObsGroup):  # 策略网络的观测组
        """给策略网络（Actor）的观测。

        这里把“参考动作命令”显式拼到观测里（motion_command），让策略知道当前应该模仿的目标关节状态。
        另外还加入 motion_anchor_ori_b 这类“相对误差”观测，帮助策略对齐方向。

        注意：这里启用了 `enable_corruption=True`，即对部分观测加入噪声（更鲁棒）。
        """

        # observation terms (order preserved) 观测项（顺序保留）
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})  # 当前时间步的模仿命令
        motion_anchor_ori_b = ObsTerm(  # 锚心相对姿态与目标的误差
            func=mdp.motion_anchor_ori_b,  # 对应的观测方法
            params={"command_name": "motion"},  # 输入参数：命令名
            noise=Unoise(n_min=-0.05, n_max=0.05)  # 加入均匀噪声
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))  # 机器人基座角速度并加噪
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))  # 关节相对角度并加噪
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))  # 关节相对速度并加噪
        last_action = ObsTerm(func=mdp.last_action)  # 上一次的动作

        def __post_init__(self):  # 初始化后自动执行
            self.enable_corruption = True  # 启用观测扰动
            self.concatenate_terms = True  # 自动拼接所有观测项

    @configclass
    class PrivilegedCfg(ObsGroup):  # 价值网络特权观测组
        """给价值网络（Critic）的特权观测（Privileged）。

        Critic 通常可以看到更多真实状态（例如 base_lin_vel、机器人身体姿态等），
        这能让值函数更准确，进而提升 PPO 训练稳定性。
        """
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})  # 当前时间步的命令
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})  # 锚心相对位置
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})  # 锚心相对姿态
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})  # 身体各部位相对目标位置
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})  # 身体各部位相对目标方向
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # 机器人基座线速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)  # 基座角速度
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)  # 关节相对角度
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)  # 关节相对速度
        actions = ObsTerm(func=mdp.last_action)  # 上一时刻的动作

    # observation groups 观测组
    policy: PolicyCfg = PolicyCfg()  # 策略网络观测组
    critic: PrivilegedCfg = PrivilegedCfg()  # 价值网络观测组


@configclass
class EventCfg:  # 域随机化及扰动事件配置
    """环境事件（域随机化/扰动）配置。

    这类配置的主要目的不是“让任务更难”，而是让策略学到在真实世界也可用的鲁棒性：
    - 摩擦随机：模拟不同地面材质
    - 关节零位随机：模拟标定误差（非常贴近真机）
    - CoM 随机：模拟背包/电池位置变化
    - push_robot：定期随机速度扰动，测试恢复能力
    """

    # startup 初始化阶段
    physics_material = EventTerm(  # 随机地面摩擦参数
        func=mdp.randomize_rigid_body_material,  # 执行的方法
        mode="startup",  # 只在仿真启动时触发
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 作用对象为所有link
            "static_friction_range": (0.3, 1.6),  # 静摩擦系数随机范围
            "dynamic_friction_range": (0.3, 1.2),  # 动摩擦系数随机范围
            "restitution_range": (0.0, 0.5),  # 弹性回复系数随机范围
            "num_buckets": 64,  # 随机采样的区间数
        },
    )

    add_joint_default_pos = EventTerm(  # 关节初始零位扰动
        func=mdp.randomize_joint_default_pos,  # 方法
        mode="startup",  # 初始化时触发
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),  # 作用所有关节
            "pos_distribution_params": (-0.01, 0.01),  # 角度扰动范围
            "operation": "add",  # 操作类型为“加”
        },
    )

    base_com = EventTerm(  # 基座质心扰动
        func=mdp.randomize_rigid_body_com,  # 随机质心位置
        mode="startup",  # 初始化随机
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),  # 仅作用于torso_link
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},  # 质心扰动范围
        },
    )

    # interval 定期触发
    push_robot = EventTerm(  # 随机外推力扰动
        func=mdp.push_by_setting_velocity,  # 设置速度的“推”操作
        mode="interval",  # 定期触发
        interval_range_s=(1.0, 3.0),  # 1到3秒间隔
        params={"velocity_range": VELOCITY_RANGE},  # 速度扰动范围
    )


@configclass
class RewardsCfg:  # 奖励配置
    """奖励项配置。

    Mimic 的奖励大致分两类：
    - **tracking（跟踪）**：让机器人尽量贴近参考动作（位置/姿态/速度）
    - **regularization（正则）**：抑制过大力矩、关节加速度、动作突变、越界等
    """

    # -- base 正则项
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # 关节加速度L2正则罚项
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)  # 关节力矩L2正则罚项
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)  # 动作变化幅度罚项
    joint_limit = RewTerm(  # 关节越界惩罚
        func=mdp.joint_pos_limits,  # 关节角限制检测方法
        weight=-10.0,  # 罚项权重
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},  # 所有关节
    )

    # -- tracking 模仿项
    motion_global_anchor_pos = RewTerm(  # 锚心位置与目标误差
        func=mdp.motion_global_anchor_position_error_exp,  # 计算误差的指数函数
        weight=0.5,  # 权重
        params={"command_name": "motion", "std": 0.3},  # 输入参数
    )
    motion_global_anchor_ori = RewTerm(  # 锚心姿态与目标误差
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(  # 各身体位置与目标误差
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(  # 各身体姿态与目标误差
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(  # 各身体线速度误差
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(  # 各身体角速度误差
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )

    undesired_contacts = RewTerm(  # 不希望的接触惩罚
        func=mdp.undesired_contacts,
        weight=-0.1,  # 权重为负
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",  # 传感器资产名
                body_names=[
                    # 排除合法的脚踝、手腕，其余接触惩罚
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,  # 力阈值
        },
    )


@configclass
class TerminationsCfg:  # Done条件配置
    """终止条件（Done）配置。

    Mimic 任务常见的失败原因是：
    - anchor（通常是 torso_link）高度/姿态偏差过大：整体跟丢参考
    - 末端（脚踝、手腕）高度偏差过大：动作局部崩坏
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # 达到最大步数超时终止
    anchor_pos = DoneTerm(  # 锚心z方向偏差过大终止
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(  # 锚心姿态偏差过大终止
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(  # 末端（脚踝/手腕）z方向偏差过大终止
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )


##
# Environment configuration 环境总配置
##


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):  # 基于管理器的强化学习环境总配置
    """Dance-102 Mimic 环境配置（训练用）。"""

    # Scene settings 场景设置
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)  # 每次并行4096个环境，间距2.5
    # Basic settings 基本设置
    observations: ObservationsCfg = ObservationsCfg()  # 观测空间
    actions: ActionsCfg = ActionsCfg()  # 动作空间
    commands: CommandsCfg = CommandsCfg()  # 命令
    # MDP settings MDP配置
    rewards: RewardsCfg = RewardsCfg()  # 奖励函数
    terminations: TerminationsCfg = TerminationsCfg()  # 终止条件
    events: EventCfg = EventCfg()  # 环境事件
    curriculum = None  # 未启用课程学习

    def __post_init__(self):  # 初始化后自动执行
        """Post initialization."""
        # general settings 一般配置
        self.decimation = 4  # 动作重复步数
        self.episode_length_s = 30.0  # 回合最大时长（秒）
        # simulation settings 仿真相关
        self.sim.dt = 0.005  # 仿真基本步长
        self.sim.render_interval = self.decimation  # 渲染间隔（每几个仿真步渲染一次）
        self.sim.physics_material = self.scene.terrain.physics_material  # 设置仿真物理材质
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15  # PhysX物理的GPU刚体补丁上限


class RobotPlayEnvCfg(RobotEnvCfg):  # 播放完整动作序列用的环境配置
    def __post_init__(self):  # 初始化后自动执行
        super().__post_init__()  # 调用父类初始化
        self.scene.num_envs = 1  # 只开一个环境用于展示
        self.episode_length_s = 1e9  # 非常长的回合时长以完整播放序列
