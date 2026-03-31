# Copyright (c) 2022-2025, The Isaac Lab Project Developers.  # 版权声明，注明代码所属及年份
# All rights reserved.                                       # 所有权声明
#
# SPDX-License-Identifier: BSD-3-Clause                      # 软件许可协议声明

"""Configuration for Unitree robots.                          # Unitree机器人配置文件说明

Reference: https://github.com/unitreerobotics/unitree_ros    # 参考链接，指向Unitree官方ROS包
"""

import os                                                     # 导入os模块，用于文件/目录操作

import isaaclab.sim as sim_utils                              # 导入Isaac Lab的仿真工具并重命名为sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg  # 导入理想PD和隐式执行器配置
from isaaclab.assets.articulation import ArticulationCfg      # 导入关节结构配置
from isaaclab.utils import configclass                        # 导入配置类装饰器

from unitree_rl_lab.assets.robots import unitree_actuators    # 导入自定义Unitree关节驱动配置

UNITREE_MODEL_DIR = "/home/dex/andy/source/code/unitree_model"  # Unitree模型路径，请替换为真实路径
UNITREE_ROS_DIR = "/home/dex/andy/source/code/unitree_ros"      # Unitree ROS包路径，请替换为真实路径


@configclass
class UnitreeArticulationCfg(ArticulationCfg):                # 定义Unitree扩展的关节结构配置类
    """Configuration for Unitree articulations."""            # 类注释（英文）

    joint_sdk_names: list[str] = None                        # SDK关节名称（用于与真实机器人协议对接）

    soft_joint_pos_limit_factor = 0.9                        # 软关节位置限制系数（安全范围百分比）


@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):                # 定义USD模型文件配置类，继承自UsdFileCfg
    activate_contact_sensors: bool = True                    # 是否激活接触传感器
    rigid_props = sim_utils.RigidBodyPropertiesCfg(           # 刚体物理属性配置
        disable_gravity=False,                               # 是否关闭重力（False表示有重力）
        retain_accelerations=False,                          # 是否保留加速度信息
        linear_damping=0.0,                                  # 线性阻尼
        angular_damping=0.0,                                 # 角阻尼
        max_linear_velocity=1000.0,                          # 最大线速度
        max_angular_velocity=1000.0,                         # 最大角速度
        max_depenetration_velocity=1.0,                      # 最大穿透校正速度
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(  # 关节根配置
        enabled_self_collisions=True,                        # 启用自碰撞
        solver_position_iteration_count=8,                   # 位置迭代次数
        solver_velocity_iteration_count=4                    # 速度迭代次数
    )


@configclass
class UnitreeUrdfFileCfg(sim_utils.UrdfFileCfg):              # 定义URDF模型文件配置类
    fix_base: bool = False                                   # 是否锁定基座（默认不锁定）
    activate_contact_sensors: bool = True                    # 激活接触传感器
    replace_cylinders_with_capsules = True                   # 是否用胶囊体替换圆柱体，提高物理仿真稳定性
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(  # 关节驱动参数配置
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)  # PD参数为0（默认）
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(  # 关节根属性
        enabled_self_collisions=True,                        # 开启自碰撞检测
        solver_position_iteration_count=8,                   # 位置求解器迭代次数
        solver_velocity_iteration_count=4,                   # 速度求解器迭代次数
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(           # 刚体物理属性配置
        disable_gravity=False,                               # 不关闭重力
        retain_accelerations=False,                          # 不保留加速度
        linear_damping=0.0,                                  # 线性阻尼
        angular_damping=0.0,                                 # 角阻尼
        max_linear_velocity=1000.0,                          # 最大线速度
        max_angular_velocity=1000.0,                         # 最大角速度
        max_depenetration_velocity=1.0,                      # 最大穿透速度
    )

    def replace_asset(self, meshes_dir, urdf_path):          # 用于用符号链接方式替换URDF和mesh，避免更改原始模型
        """Replace the asset with a temporary copy to avoid modifying the original asset.

        When need to change the collisions, place the modified URDF file separately in this repository,
        and let `meshes_dir` be provided by `unitree_ros`.
        This function will auto construct a complete `robot_description` file structure in the `/tmp` directory.
        Note: The mesh references inside the URDF should be in the same directory level as the URDF itself.
        """
        tmp_meshes_dir = "/tmp/IsaacLab/unitree_rl_lab/meshes"   # 临时mesh目录路径
        if os.path.exists(tmp_meshes_dir):                       # 如果临时mesh存在，则删除
            os.remove(tmp_meshes_dir)
        os.makedirs("/tmp/IsaacLab/unitree_rl_lab", exist_ok=True)  # 创建临时工作目录
        os.symlink(meshes_dir, tmp_meshes_dir)                   # 创建mesh符号链接

        self.asset_path = "/tmp/IsaacLab/unitree_rl_lab/robot.urdf" # 设定临时urdf文件路径
        if os.path.exists(self.asset_path):                      # 存在时先删除
            os.remove(self.asset_path)
        os.symlink(urdf_path, self.asset_path)                   # 创建urdf文件符号链接


""" Configuration for the Unitree robots."""                        # 全局配置说明字符串

UNITREE_GO2_CFG = UnitreeArticulationCfg(                     # GO2机器人模型配置
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/go2_description/urdf/go2_description.urdf",  # URDF路径示例（注释掉）
    # ),
    spawn=UnitreeUsdFileCfg(                                  # USD模型加载路径
        usd_path=f"{UNITREE_MODEL_DIR}/Go2/usd/go2.usd",      # 指定USD文件路径
    ),
    init_state=ArticulationCfg.InitialStateCfg(               # 初始状态配置
        pos=(0.0, 0.0, 0.4),                                 # 初始位置
        joint_pos={                                           # 各关节初始角度设置
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},                                # 所有关节初始速度为0
    ),
    actuators={                                               # 执行器配置
        "GO2HV": unitree_actuators.UnitreeActuatorCfg_Go2HV(
            joint_names_expr=[".*"],                          # 选择所有关节
            stiffness=25.0,                                   # 刚度
            damping=0.5,                                      # 阻尼
            friction=0.01,                                    # 摩擦
        ),
    },
    # fmt: off
    joint_sdk_names=[                                         # SDK协议下的关节名称顺序
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
    ],
    # fmt: on
)

UNITREE_GO2W_CFG = UnitreeArticulationCfg(                    # GO2W轮式机器人配置
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/go2w_description/urdf/go2w_description.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/Go2W/usd/go2w.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness={
                ".*_hip_.*": 25.0,
                ".*_thigh_.*": 25.0,
                ".*_calf_.*": 25.0,
                ".*_foot_.*": 0,
            },
            damping=0.5,
            friction=0.01,
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"
    ],
    # fmt: on
)

UNITREE_B2_CFG = UnitreeArticulationCfg(                      # B2机器人配置
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/b2_description/urdf/b2_description.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/B2/usd/b2.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "M107-24-2": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_thigh_.*"],
            effort_limit=200,
            velocity_limit=23,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
        ),
        "2": IdealPDActuatorCfg(
            joint_names_expr=[".*_calf_.*"],
            effort_limit=320,
            velocity_limit=14,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
        ),
    },
    joint_sdk_names=UNITREE_GO2_CFG.joint_sdk_names.copy(),   # 复用GO2的关节编号顺序
)

UNITREE_H1_CFG = UnitreeArticulationCfg(                      # H1人形机器人配置
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/h1_description/urdf/h1.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/H1/h1/usd/h1.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.1),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.20,
            ".*_elbow_joint": 0.32,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV-1": IdealPDActuatorCfg(
            joint_names_expr=[".*ankle.*", ".*_shoulder_pitch_.*", ".*_shoulder_roll_.*"],
            effort_limit=40,
            velocity_limit=9,
            stiffness={
                ".*ankle.*": 40.0,
                ".*_shoulder_.*": 100.0,
            },
            damping=2.0,
            armature=0.01,
        ),
        "GO2HV-2": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_yaw_.*", ".*_elbow_.*"],
            effort_limit=18,
            velocity_limit=20,
            stiffness=50,
            damping=2.0,
            armature=0.01,
        ),
        "M107-24-1": IdealPDActuatorCfg(
            joint_names_expr=[".*_knee_.*"],
            effort_limit=300.0,
            velocity_limit=14.0,
            stiffness=200.0,
            damping=4.0,
            armature=0.01,
        ),
        "M107-24-2": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*", "torso_joint"],
            effort_limit=200,
            velocity_limit=23.0,
            stiffness={
                ".*_hip_.*": 150.0,
                "torso_joint": 300.0,
            },
            damping={
                ".*_hip_.*": 2.0,
                "torso_joint": 6.0,
            },
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "torso_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "",
        "left_ankle_joint",
        "right_ankle_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
    ],
)

UNITREE_G1_23DOF_CFG = UnitreeArticulationCfg(                # G1 23自由度模型配置
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/g1_description/g1_23dof_rev_1_0.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/G1/23dof/usd/g1_23dof_rev_1_0/g1_23dof_rev_1_0.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],  # 5
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            stiffness={
                ".*_hip_.*": 100.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                ".*_hip_.*": 2.0,
                "waist_yaw_joint": 5.0,
            },
            armature=0.01,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],  # 4
            effort_limit_sim=139,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_roll_.*": 100.0,
                ".*_knee_.*": 150.0,
            },
            damping={
                ".*_hip_roll_.*": 2.0,
                ".*_knee_.*": 4.0,
            },
            armature=0.01,
        ),
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_roll_.*"],  # 10
            effort_limit_sim=25,
            velocity_limit_sim=37,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
        "N5020-16-parallel": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle.*"],  # 4
            effort_limit_sim=35,
            velocity_limit_sim=30,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "",
        "",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "",
        "",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
    ],
)

UNITREE_G1_29DOF_CFG = UnitreeArticulationCfg(  # 创建G1 29自由度模型的关节配置对象
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/g1_description/g1_29dof_rev_1_0.urdf",
    # ),  # 可选地支持从URDF模型加载，当前使用USD模型
    spawn=UnitreeUsdFileCfg(  # 指定仿真时加载的USD模型文件配置
        usd_path=f"{UNITREE_MODEL_DIR}/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",  # USD模型路径
    ),
    init_state=ArticulationCfg.InitialStateCfg(  # 初始化状态设置
        pos=(0.0, 0.0, 0.8),  # 初始位置，[x, y, z]，z较大保证站立
        joint_pos={  # 各关节的初始角度配置
            "left_hip_pitch_joint": -0.1,  # 左髋前后关节初始角度
            "right_hip_pitch_joint": -0.1,  # 右髋前后关节初始角度
            ".*_knee_joint": 0.3,  # 所有膝关节的初始弯曲角
            ".*_ankle_pitch_joint": -0.2,  # 所有踝关节pitch分量的初始角度
            ".*_shoulder_pitch_joint": 0.3,  # 所有肩部pitch分量的初始角度
            "left_shoulder_roll_joint": 0.25,  # 左肩roll关节初始角度
            "right_shoulder_roll_joint": -0.25,  # 右肩roll关节初始角度
            ".*_elbow_joint": 0.97,  # 所有肘关节的初始角度
            "left_wrist_roll_joint": 0.15,  # 左手腕roll关节初始角度
            "right_wrist_roll_joint": -0.15,  # 右手腕roll关节初始角度
        },
        joint_vel={".*": 0.0},  # 所有关节初始速度设置为0
    ),
    actuators={  # 设置各类型驱动电机和匹配关节
        "N7520-14.3": ImplicitActuatorCfg(  # N7520-14.3型号驱动器配置
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],  # 匹配髋pitch与yaw以及腰部yaw关节
            effort_limit_sim=88,  # 模拟中此类电机允许的最大力矩
            velocity_limit_sim=32.0,  # 最大角速度
            stiffness={  # 刚度配置
                ".*_hip_.*": 100.0,  # 髋相关关节的刚度
                "waist_yaw_joint": 200.0,  # 腰部yaw的刚度
            },
            damping={  # 阻尼配置
                ".*_hip_.*": 2.0,  # 髋相关关节的阻尼
                "waist_yaw_joint": 5.0,  # 腰部yaw的阻尼
            },
            armature=0.01,  # 转动惯量设定
        ),
        "N7520-22.5": ImplicitActuatorCfg(  # N7520-22.5型号驱动器配置
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],  # 匹配髋roll与膝关节
            effort_limit_sim=139,  # 最大可用力矩（模拟值）
            velocity_limit_sim=20.0,  # 最大角速度（模拟值）
            stiffness={  # 刚度设置
                ".*_hip_roll_.*": 100.0,  # 髋roll刚度
                ".*_knee_.*": 150.0,  # 膝盖刚度
            },
            damping={  # 阻尼设置
                ".*_hip_roll_.*": 2.0,  # 髋roll阻尼
                ".*_knee_.*": 4.0,  # 膝盖阻尼
            },
            armature=0.01,  # 转动惯量
        ),
        "N5020-16": ImplicitActuatorCfg(  # N5020-16型号电机配置
            joint_names_expr=[  # 匹配的关节名称
                ".*_shoulder_.*",  # 所有肩关节
                ".*_elbow_.*",     # 所有肘关节
                ".*_wrist_roll.*", # 所有腕部roll关节
                ".*_ankle_.*",     # 所有踝关节
                "waist_roll_joint",     # 腰部roll关节
                "waist_pitch_joint",    # 腰部pitch关节
            ],
            effort_limit_sim=25,  # 力矩上限
            velocity_limit_sim=37,  # 速度上限
            stiffness=40.0,  # 刚度（统一配置）
            damping={  # 针对不同类别关节的阻尼
                ".*_shoulder_.*": 1.0,    # 肩关节阻尼
                ".*_elbow_.*": 1.0,       # 肘关节阻尼
                ".*_wrist_roll.*": 1.0,   # 腕roll阻尼
                ".*_ankle_.*": 2.0,       # 踝关节阻尼
                "waist_.*_joint": 5.0,    # 腰部全部关节
            },
            armature=0.01,  # 转动惯量
        ),
        "W4010-25": ImplicitActuatorCfg(  # W4010-25型号电机（手腕pitch、yaw）
            joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],  # 匹配所有手腕pitch和yaw
            effort_limit_sim=5,  # 最大力矩较小，适应手腕关节
            velocity_limit_sim=22,  # 速度上限
            stiffness=40.0,  # 刚度
            damping=1.0,     # 阻尼
            armature=0.01,   # 转动惯量
        ),
    },
    joint_sdk_names=[  # 与实际机器人SDK协议顺序对应的关节名数组
        "left_hip_pitch_joint",      # 左髋pitch
        "left_hip_roll_joint",       # 左髋roll
        "left_hip_yaw_joint",        # 左髋yaw
        "left_knee_joint",           # 左膝
        "left_ankle_pitch_joint",    # 左踝pitch
        "left_ankle_roll_joint",     # 左踝roll
        "right_hip_pitch_joint",     # 右髋pitch
        "right_hip_roll_joint",      # 右髋roll
        "right_hip_yaw_joint",       # 右髋yaw
        "right_knee_joint",          # 右膝
        "right_ankle_pitch_joint",   # 右踝pitch
        "right_ankle_roll_joint",    # 右踝roll
        "waist_yaw_joint",           # 腰yaw
        "waist_roll_joint",          # 腰roll
        "waist_pitch_joint",         # 腰pitch
        "left_shoulder_pitch_joint", # 左肩pitch
        "left_shoulder_roll_joint",  # 左肩roll
        "left_shoulder_yaw_joint",   # 左肩yaw
        "left_elbow_joint",          # 左臂肘关节
        "left_wrist_roll_joint",     # 左腕roll
        "left_wrist_pitch_joint",    # 左腕pitch
        "left_wrist_yaw_joint",      # 左腕yaw
        "right_shoulder_pitch_joint",# 右肩pitch
        "right_shoulder_roll_joint", # 右肩roll
        "right_shoulder_yaw_joint",  # 右肩yaw
        "right_elbow_joint",         # 右臂肘关节
        "right_wrist_roll_joint",    # 右腕roll
        "right_wrist_pitch_joint",   # 右腕pitch
        "right_wrist_yaw_joint",     # 右腕yaw
    ],  # 顺序与mimic数据及SDK协议保持一致
)


ARMATURE_5020 = 0.003609725                                  # 5020型号电机转动惯量
ARMATURE_7520_14 = 0.010177520                               # 7520-14型号转动惯量
ARMATURE_7520_22 = 0.025101925                               # 7520-22型号转动惯量
ARMATURE_4010 = 0.00425                                      # 4010型号转动惯量

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz               # 自然频率（10Hz * 2π）
DAMPING_RATIO = 2.0                                         # 阻尼比

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2            # 5020刚度，采用Jw^2公式
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2      # 7520-14刚度
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2      # 7520-22刚度
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2            # 4010刚度

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ     # 5020阻尼，采用2ζJw
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ  # 7520-14阻尼
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ  # 7520-22阻尼
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ        # 4010阻尼

UNITREE_G1_29DOF_MIMIC_CFG = UnitreeArticulationCfg(         # G1 29自由度模仿任务机器人资产配置
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/g1_description/g1_29dof_rev_1_0.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),                                # 初始位置略矮（模仿数据适配）
        joint_pos={
            ".*_hip_pitch_joint": -0.312,                    # 髋关节初始角度
            ".*_knee_joint": 0.669,                           # 膝关节初始角度
            ".*_ankle_pitch_joint": -0.363,                  # 踝关节初始角度
            ".*_elbow_joint": 0.6,                           # 肘关节初始
            "left_shoulder_roll_joint": 0.2,                 # 左肩roll
            "left_shoulder_pitch_joint": 0.2,                # 左肩pitch
            "right_shoulder_roll_joint": -0.2,               # 右肩roll
            "right_shoulder_pitch_joint": 0.2,               # 右肩pitch
        },
        joint_vel={".*": 0.0},                               # 所有关节速度0
    ),
    soft_joint_pos_limit_factor=0.9,                         # 软极限缩放
    actuators={
        "legs": ImplicitActuatorCfg(                         # 四肢驱动
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(                         # 脚部执行器
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(                        # 腰部roll、pitch
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(                    # 腰部yaw
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(                         # 上肢执行器（肩、肘、腕）
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
)

# 计算G1 29DOF仿真动作归一化的标量（用于Mimic任务action scale）
UNITREE_G1_29DOF_MIMIC_ACTION_SCALE = {}                     # 定义动作缩放字典
for a in UNITREE_G1_29DOF_MIMIC_CFG.actuators.values():      # 遍历所有执行器配置
    e = a.effort_limit_sim                                   # 获取力矩上限
    s = a.stiffness                                          # 获取刚度
    names = a.joint_names_expr                               # 关节名称表达式列表
    if not isinstance(e, dict):                              # 如果力矩上限不是dict（即不是每关节单独限制）
        e = {n: e for n in names}                            # 则为所有关节赋同一个限制
    if not isinstance(s, dict):                              # 如果刚度不是dict（同理）
        s = {n: s for n in names}
    for n in names:                                          # 遍历每个关节表达式
        if n in e and n in s and s[n]:                       # 如果该关节在上限和刚度字典且刚度不为0
            UNITREE_G1_29DOF_MIMIC_ACTION_SCALE[n] = 0.25 * e[n] / s[n]   # 动作归一化系数计算（0.25*上限/刚度）
# End of loop and code section
