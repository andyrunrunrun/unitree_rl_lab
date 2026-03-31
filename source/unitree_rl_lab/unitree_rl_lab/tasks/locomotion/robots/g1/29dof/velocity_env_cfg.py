import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# COBBLESTONE_ROAD_CFG 定义了用于仿真训练环境的地形生成器配置。
# 它指定了每个子地形块（即每个仿真 environment 落脚区域）的尺寸、地形难度层级、采样分辨率、支持的地形类型（此处仅启用 flat），
# 以及缓存与边界参数。课程学习训练过程中会根据该配置生成不同难度和多样性的局部地形块供机器人适应和行走。
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    # 单个子地形块大小（米）：训练时每个 environment 会落在该网格地形上
    size=(8.0, 8.0),
    # 地形外围额外边界宽度（米），用于避免边缘效应
    border_width=20.0,
    # 行列数决定课程学习可用的地形等级数量和横向多样性
    num_rows=9,
    num_cols=21,
    # 水平/竖直采样分辨率：影响地形几何细节精度
    horizontal_scale=0.1,
    vertical_scale=0.005,
    # 坡度阈值：用于生成和过滤可行走地形区域
    slope_threshold=0.75,
    # 难度范围：用于课程学习时从易到难抽样
    difficulty_range=(0.0, 1.0),
    # 关闭缓存，确保每次启动可重新生成（可提高随机性）
    use_cache=False,
    sub_terrains={
        # 这里只启用平地，比例 0.5；其余复杂地形类型可按需扩展
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """场景配置：包含地形、机器人本体、传感器与光照。

    该类定义了在仿真中用于训练 G1 29dof 机器人行走任务的场景要素。包括地形导入、机器人模型、传感器挂载方式、
    以及基本光照资源。用于后续 RL 环境的场景初始化与管理。
    """

    # 地面地形导入器：使用上面定义的生成式地形配置
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # 可选 "plane"（纯平面）或 "generator"（程序化地形）
        terrain_generator=COBBLESTONE_ROAD_CFG,  # 可替换为 None 或其它地形配置
        # 初始地形等级上限：这里允许初始化到最高等级（num_rows - 1）
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        # 地面物理材质：摩擦采用 multiply 组合，提升接触稳定性
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # 地面可视材质（仅渲染表现，不影响动力学）
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # 机器人实体：将机器人 prim 挂到每个并行环境的命名空间下
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 高度扫描器：从 torso_link 上方投射射线，感知局部地形起伏（当前观测中默认未启用）
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # 接触力传感器：挂在机器人所有刚体上，记录最近 3 帧接触并跟踪腾空时间
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # 场景天空光：提供基础环境照明
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """事件域随机化与扰动配置（startup/reset/interval 三类）。"""

    # startup：环境创建时执行一次，用于提升 sim-to-real 鲁棒性
    # physics_material：在环境启动阶段对机器人的所有刚体（body_names=".*"）的物理材质参数进行随机化，
    # 以增强 sim-to-real 鲁棒性。具体包括静摩擦系数、动摩擦系数在指定范围内均匀采样，恢复系数固定为 0，
    # 并以 num_buckets 离散化，用于多环境随机扰动。
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # pyright: ignore[reportAttributeAccessIssue]
        mode="startup",
        params={
            # 随机化物理材质参数时作用的目标实体（此处为机器人全身所有刚体）
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # 静摩擦系数量化采样范围
            "static_friction_range": (0.3, 1.0),
            # 动摩擦系数量化采样范围
            "dynamic_friction_range": (0.3, 1.0),
            # 恢复系数量化采样范围（此处恒为 0，表示无弹性碰撞）
            "restitution_range": (0.0, 0.0),
            # 采样离散桶数量（用于多环境批量扰动时分桶）
            "num_buckets": 64,
        },
    )

    # add_base_mass：在环境启动阶段对机器人躯干质量进行扰动，随机增加一定范围的质量（-1.0 到 3.0，支持局部负载变化），用于提升对负载不确定性的鲁棒性。
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # pyright: ignore[reportAttributeAccessIssue]
        mode="startup",
        params={
            # 仅扰动躯干质量，模拟载荷变化
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset：每个 episode 重置时执行
    # base_external_force_torque：在每个 episode 重置时，对机器人躯干（torso_link）施加外部力和力矩扰动（此处为0，默认不扰动，可用于测试对外部干扰的鲁棒性）
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque, # pyright: ignore[reportAttributeAccessIssue]
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, # pyright: ignore[reportAttributeAccessIssue]
        mode="reset",
        params={
            # 初始位姿随机范围：x/y 平移与 yaw 朝向随机
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                # 根速度初始化为 0，减少初始瞬态干扰
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale, # pyright: ignore[reportAttributeAccessIssue]
        mode="reset",
        params={
            # position_range=(1,1) 表示回到默认关节位姿；速度做一定随机扰动
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval：按固定时间间隔注入外部扰动（推搡）
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity, # pyright: ignore[reportAttributeAccessIssue]
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """MDP 指令配置：定义策略需要跟踪的目标速度分布。"""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        # 每 10s 重采样一次速度目标
        resampling_time_range=(10.0, 10.0),
        # 少量环境保持“站立”指令，避免策略只会移动不会静止
        rel_standing_envs=0.02,
        # 全部环境都参与朝向相关设置（此处 heading_command=False，仅保留接口一致性）
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        # 训练期常规采样范围（较窄，先学稳定步态）
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        # 推理/放宽限制的速度范围（更大）
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    )


@configclass
class ActionsCfg:
    """MDP 动作配置：策略输出映射为关节位置目标。"""

    JointPositionAction = mdp.JointPositionActionCfg( # pyright: ignore[reportAttributeAccessIssue]
        # 全关节位置控制；scale 控制动作幅度；use_default_offset 使用默认姿态作偏置
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """MDP 观测配置：分 policy（actor）与 critic（privileged）两组。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络观测（带噪声/扰动，更贴近真实部署输入）。"""

        # 观测项（顺序保留，最终会按顺序拼接）
        # 机器人机体（base）的角速度观测项，反映当前旋转动态。添加了归一化缩放与高斯噪声，增强鲁棒性。
        # 机器人机体（base）的角速度观测项，反映当前旋转动态。添加缩放（scale）和高斯噪声（noise）以提升训练鲁棒性。
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2) # pyright: ignore[reportAttributeAccessIssue]
        )  # pyright: ignore[reportAttributeAccessIssue]

        # 观测地心重力在机体坐标系下的投影方向，辅助机器人感知自身姿态。添加少量噪声。
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05) # pyright: ignore[reportAttributeAccessIssue]
        )  # pyright: ignore[reportAttributeAccessIssue]

        # 目标速度指令观测（即当前期望的运动速度），为策略提供任务目标。
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"} # pyright: ignore[reportAttributeAccessIssue]
        )  # pyright: ignore[reportAttributeAccessIssue]

        # 关节位置的相对观测，形式为与默认/初始姿态的偏差。带噪声，模拟实际传感器误差。
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01) # pyright: ignore[reportAttributeAccessIssue]
        )  # pyright: ignore[reportAttributeAccessIssue]

        # 关节速度的相对观测，对应实际电机传感器信息。设置缩放比例及较大幅度噪声，提高鲁棒性。
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5) # pyright: ignore[reportAttributeAccessIssue]
        )  # pyright: ignore[reportAttributeAccessIssue]

        # 上一时刻动作输出，用于引导策略学习动作平滑性及相关性。
        last_action = ObsTerm(func=mdp.last_action)  # pyright: ignore[reportAttributeAccessIssue]
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            # 堆叠最近 5 帧历史，增强时序信息表达能力
            self.history_length = 5
            # 启用观测污染（噪声等）
            self.enable_corruption = True
            # 将各观测项拼接为单个向量
            self.concatenate_terms = True

    # actor 使用的主观测组
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """价值网络观测（通常更“干净”或更全面）。"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel) # pyright: ignore[reportAttributeAccessIssue]
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2) # pyright: ignore[reportAttributeAccessIssue]
        projected_gravity = ObsTerm(func=mdp.projected_gravity) # pyright: ignore[reportAttributeAccessIssue]
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) # pyright: ignore[reportAttributeAccessIssue]
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel) # pyright: ignore[reportAttributeAccessIssue]
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05) # pyright: ignore[reportAttributeAccessIssue]
        last_action = ObsTerm(func=mdp.last_action) # pyright: ignore[reportAttributeAccessIssue]
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            # critic 同样使用短历史窗口，提升价值估计稳定性
            self.history_length = 5

    # critic（特权观测）组
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """奖励函数配置：任务跟踪奖励 + 稳定性/能耗/姿态约束惩罚。"""

    # -- 任务目标：跟踪指令速度
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, # pyright: ignore[reportAttributeAccessIssue]
        weight=1.0,
        # std 越小，对误差越敏感；指数型奖励对小误差更友好
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}, # pyright: ignore[reportAttributeAccessIssue]
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)} # pyright: ignore[reportAttributeAccessIssue]
    )

    # 存活奖励：鼓励保持不倒
    alive = RewTerm(func=mdp.is_alive, weight=0.15) # pyright: ignore[reportAttributeAccessIssue]

    # -- 底座与运动平滑性约束
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0) # pyright: ignore[reportAttributeAccessIssue]
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05) # pyright: ignore[reportAttributeAccessIssue]
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001) # pyright: ignore[reportAttributeAccessIssue]
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7) # pyright: ignore[reportAttributeAccessIssue]
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05) # pyright: ignore[reportAttributeAccessIssue]
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0) # pyright: ignore[reportAttributeAccessIssue]
    energy = RewTerm(func=mdp.energy, weight=-2e-5) # pyright: ignore[reportAttributeAccessIssue]

    # 上肢偏离默认位姿惩罚：抑制无意义摆臂
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1, # pyright: ignore[reportAttributeAccessIssue]
        weight=-0.1, # pyright: ignore[reportAttributeAccessIssue]
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    # 腰部偏离惩罚：约束躯干扭动
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1, # pyright: ignore[reportAttributeAccessIssue]
        weight=-1, # pyright: ignore[reportAttributeAccessIssue]
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    # 髋关节滚转/偏航偏离惩罚：保持下肢稳定
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1, # pyright: ignore[reportAttributeAccessIssue]
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- 机体姿态约束
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0) # pyright: ignore[reportAttributeAccessIssue]
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78}) # pyright: ignore[reportAttributeAccessIssue]

    # -- 足端步态与接触质量
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            # 双足相位参数：offset=[0.0, 0.5] 对应近似反相步态
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide, # pyright: ignore[reportAttributeAccessIssue]
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # 抬脚高度奖励：鼓励迈步时脚尖有足够离地间隙，减少绊脚
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- 其它约束：非足端身体部位接触地面视作不期望接触
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, # pyright: ignore[reportAttributeAccessIssue]
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """终止条件配置：超时、过低、姿态异常。"""

    # 正常时间结束
    time_out = DoneTerm(func=mdp.time_out, time_out=True) # pyright: ignore[reportAttributeAccessIssue]
    # 根部高度低于阈值（跌倒/蹲伏过低）终止
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2}) # pyright: ignore[reportAttributeAccessIssue]
    # 倾斜角过大终止
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8}) # pyright: ignore[reportAttributeAccessIssue]


@configclass
class CurriculumCfg:
    """课程学习项：随训练进度逐步提升任务难度。"""

    # 地形等级课程：逐步暴露更难地形
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel) # pyright: ignore[reportAttributeAccessIssue]
    # 指令速度课程：逐步放宽目标速度范围
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """速度跟踪行走环境总配置（训练版）。"""

    # 场景配置（并行环境数较大，用于高吞吐训练）
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # 基础 MDP 组件
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP 运行组件
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """后处理：统一设置仿真步长、传感器刷新和课程开关。"""
        # 通用设置
        # decimation=4 表示策略每 4 个物理步执行一次动作（控制频率 = 1/(dt*decimation)）
        self.decimation = 4
        self.episode_length_s = 20.0
        # 仿真设置
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # 让全局仿真材质与地形材质对齐，减少接触参数不一致
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # 传感器刷新周期
        # 接触传感器按物理步刷新；高度扫描器按控制步刷新（节省计算）
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # 若启用地形课程学习，则同步打开地形生成器的 curriculum 开关
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """推理/演示配置：减少并行数量并扩大速度命令范围。"""

    def __post_init__(self):
        super().__post_init__()
        # 推理时减少环境数量，降低显存与算力占用
        self.scene.num_envs = 32
        # 缩小地形网格，提升加载与交互速度
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        # 使用更宽的命令范围（limit_ranges）用于测试策略泛化
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
