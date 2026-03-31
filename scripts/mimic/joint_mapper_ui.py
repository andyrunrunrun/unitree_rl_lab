"""G1 29DOF 关节映射可视化工具（UI 滑条版）。

===============================================================================
这个脚本解决什么问题？
===============================================================================
初学者在看 `UNITREE_G1_29DOF_CFG.joint_sdk_names` 时，常见困惑是：
- 名字知道了，但不知道“动这个关节到底会影响身体哪一块”；
- 想快速验证左右腿/腰/手臂关节是否理解正确；
- 想把某个姿态手动调出来，观察动作组合效果。

这个工具就是把以上过程可视化：
1) 启动 Isaac Sim；
2) 在场景里放一台 G1；
3) 用 29 个滑条（按 sdk 顺序）映射 29 个关节；
4) 实时把滑条角度写入机器人关节。

===============================================================================
运行方式
===============================================================================
    python scripts/mimic/joint_mapper_ui.py

可选参数（由 AppLauncher 注入）：
- `--device cuda:0/cpu`：选择仿真设备
- `--headless`：无窗口模式（注意：UI 模式一般不建议 headless）

===============================================================================
实现思路（高层）
===============================================================================
`UI current_angles` 作为单一“真值源”：
- 每次滑条变化 -> 更新 `current_angles`
- 每帧仿真循环 -> 把 `current_angles` 写回机器人关节

这里采用“教学演示模式”：
- 不执行 `sim.step()` 物理积分；
- 直接 `write_joint_state_to_sim(...)` + `render()`。

优点是：拖动滑条时响应快、直观、不受动力学漂移影响。
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visual UI mapper for G1 29DOF joints.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 注意：AppLauncher 必须在导入大量 Isaac Sim/Omniverse 运行时模块之前创建，
# 否则会因为运行时上下文尚未初始化而导入失败。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG


@configclass
class JointMapperSceneCfg(InteractiveSceneCfg):
    """地面 + 灯光 + 一台 G1 的最小教学场景。

    只保留必要元素：
    - `ground`：提供视觉参考（看得清“脚在地面上的姿态”）
    - `sky_light`：避免模型过暗/全黑
    - `robot`：目标 G1 资产
    """

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class JointMapperUI:
    """构建并维护关节滑条界面。

    设计原则：
    - UI 只做两件事：展示 + 修改 `current_angles`
    - 不直接操作仿真对象，避免 UI 与仿真逻辑耦合
    - 通过 `current_angles` 与外部主循环通信
    """

    def __init__(self, sdk_joint_names: list[str], default_angles: list[float]):
        # Isaac Sim 启动后可用 omni.ui
        import omni.ui as ui

        self.ui = ui
        # `sdk_joint_names` 与 `default_angles` 保持同一索引语义：
        #   idx i -> 第 i 个 sdk 关节
        self.sdk_joint_names = sdk_joint_names
        # 默认角度（用于 Reset）；做切片拷贝避免外部引用修改。
        self.default_angles = default_angles[:]
        # 当前角度（UI 的核心状态）。
        self.current_angles = default_angles[:]
        # 保存每个滑条对象，便于在 Reset/Zero 时批量刷新 UI。
        self._sliders: list[ui.FloatSlider] = []

        # 独立窗口，避免和主 viewport 混在一起。
        self.window = ui.Window("G1 Joint Mapper UI", width=460, height=920)
        self.window.visible = True
        self._build()

    def _build(self) -> None:
        """构建 UI 控件树。

        布局结构：
        - 标题区
        - 功能按钮区（Reset/Zero/Print）
        - 可滚动滑条区（29 行）
        """
        ui = self.ui
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label("G1 29DOF Joint Mapper", height=24)
                ui.Label("Drag sliders to change joint angles (rad)", height=18)

                with ui.HStack(height=28, spacing=8):
                    ui.Button("Reset Default", clicked_fn=self._reset_default)
                    ui.Button("Zero All", clicked_fn=self._zero_all)
                    ui.Button("Print Pose", clicked_fn=self._print_pose)

                ui.Spacer(height=4)
                with ui.ScrollingFrame(height=820):
                    with ui.VStack(spacing=4):
                        for idx, name in enumerate(self.sdk_joint_names):
                            with ui.HStack(height=24):
                                # 左侧标签显示：索引 + 关节名
                                ui.Label(f"{idx:02d} {name}", width=285)
                                # 角度范围给一个较宽泛的教学区间 [-2.8, 2.8] rad。
                                # 这不是“严格关节限位”，只是方便观察。
                                slider = ui.FloatSlider(min=-2.8, max=2.8, step=0.001, width=140)
                                slider.model.set_value(float(self.current_angles[idx]))
                                # 每个滑条绑定自己的回调，写入 current_angles[idx]。
                                slider.model.add_value_changed_fn(self._make_slider_cb(idx))
                                self._sliders.append(slider)

    def _make_slider_cb(self, idx: int):
        """为第 idx 个滑条创建回调函数（闭包）。"""

        def _cb(model):
            # UI 层只更新状态，不做任何仿真调用。
            self.current_angles[idx] = float(model.get_value_as_float())

        return _cb

    def _sync_sliders_from_angles(self) -> None:
        """把 current_angles 的值反向同步到滑条控件。

        用于按钮触发的批量状态更新（如 Reset/Zero）后刷新 UI。
        """
        for i, slider in enumerate(self._sliders):
            slider.model.set_value(float(self.current_angles[i]))

    def _reset_default(self) -> None:
        """恢复为机器人默认姿态。"""
        self.current_angles = self.default_angles[:]
        self._sync_sliders_from_angles()
        print("[UI] 已恢复默认姿态。")

    def _zero_all(self) -> None:
        """全部关节置 0。"""
        self.current_angles = [0.0] * len(self.current_angles)
        self._sync_sliders_from_angles()
        print("[UI] 已将全部关节置零。")

    def _print_pose(self) -> None:
        """把当前姿态打印到终端（方便记录/拷贝）。"""
        print("\n[当前 29 个关节角]")
        for i, n in enumerate(self.sdk_joint_names):
            print(f"  {i:02d} {n:<28} = {self.current_angles[i]: .6f} rad")


def run() -> None:
    """主流程：初始化仿真 -> 构建映射 -> 创建 UI -> 渲染循环。"""

    # 这里 dt=0.02 主要用于 scene/update 的时间推进节奏；
    # 因为我们不做物理积分，dt 不会像训练那样显著影响动力学结果。
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.02)
    sim = SimulationContext(sim_cfg)
    scene_cfg = JointMapperSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # 取出场景里的机器人对象（Articulation）。
    robot: Articulation = scene["robot"]
    # 单环境运行，env id 固定是 0。
    env_ids = torch.tensor([0], dtype=torch.long, device=sim.device)

    # 来自资产配置的“SDK 顺序关节名”。
    # 这个顺序是我们做映射学习时最关心的顺序。
    sdk_joint_names = list(UNITREE_G1_29DOF_CFG.joint_sdk_names)

    # 建立名称 -> 仿真关节索引（sim id）的映射。
    # 原因：UI 用 sdk 顺序，但底层写关节状态需要 sim 内部索引。
    joint_name_to_sim_id: dict[str, int] = {}
    for name in sdk_joint_names:
        joint_ids, _ = robot.find_joints(name)
        if len(joint_ids) == 0:
            raise RuntimeError(f"无法在仿真关节中找到: {name}")
        joint_name_to_sim_id[name] = int(joint_ids[0])

    # 记录默认 root/joint 状态：
    # - root 固定不动，保证相机与模型位置稳定
    # - joint_pos 作为每帧写入模板（先 clone 再覆盖目标关节）
    base_root_state = robot.data.default_root_state.clone()
    base_joint_pos = robot.data.default_joint_pos.clone()
    # 速度固定为 0，避免出现“写入角度同时附带速度”的干扰。
    base_joint_vel = torch.zeros_like(robot.data.default_joint_vel)

    # 把默认姿态按 sdk 顺序取出来，作为 UI 初始值。
    default_angles = [float(base_joint_pos[0, joint_name_to_sim_id[n]].item()) for n in sdk_joint_names]

    print("\n[G1 29DOF joint_sdk_names 映射]")
    for i, n in enumerate(sdk_joint_names):
        print(f"  {i:02d}: {n}")

    # 创建 UI（此后用户交互会实时更新 ui_state.current_angles）。
    ui_state = JointMapperUI(sdk_joint_names=sdk_joint_names, default_angles=default_angles)

    # 把视角固定到机器人附近，避免默认相机看向空场景导致“黑屏”观感。
    # look_at 大致对准躯干高度（~0.85m）。
    look_at = np.array([0.0, 0.0, 0.85], dtype=np.float32)
    cam_pos = look_at + np.array([2.2, 2.2, 1.0], dtype=np.float32)
    sim.set_camera_view(cam_pos, look_at)

    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        # 先从默认关节向量拷贝一份，再覆盖 sdk 关节的目标角度。
        # 这样可以避免误改非 sdk 关节或遗留上帧脏数据。
        joint_pos = base_joint_pos.clone()
        for i, name in enumerate(sdk_joint_names):
            sim_id = joint_name_to_sim_id[name]
            joint_pos[0, sim_id] = ui_state.current_angles[i]

        # 教学演示模式：直接写状态 + 渲染，不做动力学积分。
        # 解释：
        # - write_root_state_to_sim：固定机器人根姿态
        # - write_joint_state_to_sim：把当前滑条角度写进关节
        # - scene.write_data_to_sim：提交给仿真后端
        robot.write_root_state_to_sim(base_root_state, env_ids=env_ids)
        robot.write_joint_state_to_sim(joint_pos, base_joint_vel, env_ids=env_ids)
        scene.write_data_to_sim()
        # 每帧更新一次相机，防止用户误操作后看不到机器人。
        sim.set_camera_view(cam_pos, look_at)
        sim.render()
        # 更新 scene 内部缓存（传感器/数据视图等）。
        scene.update(sim_dt)

    print("[INFO] UI joint mapper 已退出。")


if __name__ == "__main__":
    run()
    simulation_app.close()
