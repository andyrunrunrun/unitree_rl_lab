"""从 CSV 回放动作到 Isaac Sim，并将仿真中记录的状态导出为 NPZ。

本脚本用于 mimic 数据准备：CSV 提供根节点与关节的期望值，在仿真里做**运动学回放**
（不执行物理步进 `sim.step()`），然后从 `robot.data` 采样完整关节与刚体位姿/速度。

使用示例::

    python csv_to_npz.py -f path_to_input.csv --input_fps 60 --output_fps 50

================================================================================
一、输入 CSV 格式（逗号分隔，一行一帧，无表头）
================================================================================

每行列顺序固定为::

    [px, py, pz, qx, qy, qz, qw, j0, j1, ..., j(N-1)]

- **px, py, pz** (3 列): 机器人根（浮动基）在世界坐标系中的平移，单位米。
- **qx, qy, qz, qw** (4 列): 根姿态四元数，**文件内为 XYZW 顺序**；读入后脚本会转换为
  Isaac Lab 使用的 **WXYZ**。
- **j0 … j(N-1)** (N 列): 关节角，单位弧度；顺序必须与当前机器人配置
  `UNITREE_G1_29DOF_CFG.joint_sdk_names` 一致（本项目为 **G1 29DOF，N=29**），
  故每行共 **3 + 4 + 29 = 36** 列。

时间轴：相邻两行的时间间隔由 ``--input_fps`` 决定，即 Δt = 1 / input_fps。

可选 ``--frame_range START END``：仅加载第 START 到 END 行（**从 1 起算，含端点**）。

================================================================================
二、输出 NPZ 格式（numpy.savez，多个数组键）
================================================================================

首次完整播放整条重采样轨迹后写入一次。各键含义与典型形状（单环境 env 0）::

    fps             标量或小数组，值为 ``--output_fps``（输出时间轴帧率）。
    joint_pos       shape (T, J)   — T 为输出帧数；J 为仿真中整机关节数（Articulation 关节维度）。
    joint_vel       shape (T, J)   — 关节角速度。
    body_pos_w      shape (T, B, 3) — 各刚体世界系位置；B 为 body 数量，顺序与 Isaac 资产一致。
    body_quat_w     shape (T, B, 4) — 各刚体世界系四元数（Isaac Lab 约定，一般为 wxyz）。
    body_lin_vel_w  shape (T, B, 3) — 各刚体世界系线速度。
    body_ang_vel_w  shape (T, B, 3) — 各刚体世界系角速度。

其中 T 等于重采样后的 ``output_frames``。关节/连杆索引与 USD 模型及 Isaac Lab 的
`robot.data` 定义一致；训练侧应按项目内文档与配置对齐索引，勿与 CSV 列顺序混用
（CSV 只覆盖 SDK 名称子集对应的关节写入，但 NPZ 的 joint_* 为全关节向量）。

================================================================================
"""

"""必须先启动 Isaac Sim 应用，再导入依赖 Isaac 运行时的模块。"""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# 命令行：输入轨迹、帧率与输出路径；其余由 AppLauncher 追加（如 headless、device）
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", "-f", type=str, required=True, help="The path to the input motion csv file.")
parser.add_argument("--input_fps", type=int, default=60, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.output_name:
    # 未指定输出路径时：与 CSV 同目录，扩展名改为 .npz
    args_cli.output_name = (
        "/".join(args_cli.input_file.split("/")[:-1]) + "/" + args_cli.input_file.split("/")[-1].replace(".csv", ".npz")
    )


# 启动 Omniverse / Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下为依赖仿真已启动的导入。"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# 预定义机器人：当前脚本仅使用 G1 29DOF 配置（关节列顺序与 joint_sdk_names 一致）
##
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """回放场景：地面、环境光、一台 G1 机器人。"""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    """从 CSV 加载轨迹，按 output_fps 重采样，并数值计算线速度、关节角速度、根角速度。"""

    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0  # 当前播放到重采样轨迹的第几帧
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """读取 CSV 为浮点张量，并拆成根平移、根四元数（转 wxyz）、关节角。"""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            # frame_range 为 1-based 行号：skiprows 跳过 START-1 行，max_rows 读连续行数
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        # CSV 中为 XYZW，Isaac 根状态使用 WXYZ
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        # 时长按首尾帧间隔计：共 (N-1) 个输入时间步
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """将输入帧均匀映射到 [0, duration)，按 output_dt 采样；位置/关节线性插值，姿态球面插值。"""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """向量/关节角的逐分量线性插值：result = (1-blend)*a + blend*b。"""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """四元数球面插值；blend 与 times 同长度，逐帧独立 slerp。"""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """由输出时间 t 映射到输入帧索引 index_0、index_1 及插值系数 blend ∈ [0,1]。"""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """对重采样后的序列求导：平移与关节用 torch.gradient；根角速度用 SO(3) 差分近似。"""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """由四元数序列估计角速度 ω（世界系下近似，与相邻帧差分步长 2*dt 配合）。

        Args:
            rotations: (T, 4) wxyz 四元数。
            dt: 输出时间步长（与 motion 序列一致）。

        Returns:
            (T, 3) 每帧角速度；首尾帧由相邻值重复填充以保持长度与轨迹一致。
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """返回当前帧的根位姿、根线/角速度、关节位置/速度；步进索引，播完一圈则复位并置 reset_flag。"""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag  # pyright: ignore[reportReturnType]


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """主循环：每帧用轨迹覆盖根与关节状态，渲染并更新场景；攒齐一整圈后写入 NPZ。"""
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    robot = scene["robot"]
    # 将 CSV 关节列按名称映射到仿真关节向量中的下标（顺序与 joint_sdk_names 一致）
    robot_joint_indexes = robot.find_joints(scene.cfg.robot.joint_sdk_names, preserve_order=True)[0]

    # 仅在第一遍完整播放期间记录；避免因窗口关闭导致列表不全时误保存
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False

    while simulation_app.is_running():
        (  # pyright: ignore[reportAssignmentType]
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # 根状态：default_root_state 提供 batch 形状，前 7 维为 pos+quat，接着线速度、角速度
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]  # 多环境时加上环境原点 XY
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # 关节：整向量先拷贝默认值，再把 SDK 关节位赋值并写入仿真
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # 不做物理积分，仅渲染；照旧调用 scene.update 以刷新资产数据缓存
        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            np.savez(args_cli.output_name, **log)
            print("[INFO]: Motion npz file saved to", args_cli.output_name)


def main():
    """创建仿真上下文与场景，reset 后进入回放循环。"""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
