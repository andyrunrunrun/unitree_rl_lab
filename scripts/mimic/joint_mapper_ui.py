"""G1 29DOF 关节映射可视化工具（UI 滑条版）。

相比 `joint_mapper.py` 的命令行输入，这个版本提供了更直观的界面：
- 每个关节一个滑条（按 `joint_sdk_names` 顺序）
- 实时拖动即可看到机器人姿态变化
- 支持 Reset（回默认姿态）和 Zero（全部置零）

运行：
    python scripts/mimic/joint_mapper_ui.py
"""

from __future__ import annotations

import argparse

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visual UI mapper for G1 29DOF joints.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
    """地面 + 灯光 + 一台 G1。"""

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
    """构建并维护关节滑条界面。"""

    def __init__(self, sdk_joint_names: list[str], default_angles: list[float]):
        # Isaac Sim 启动后可用 omni.ui
        import omni.ui as ui

        self.ui = ui
        self.sdk_joint_names = sdk_joint_names
        self.default_angles = default_angles[:]
        self.current_angles = default_angles[:]
        self._sliders: list[ui.FloatSlider] = []

        self.window = ui.Window("G1 Joint Mapper UI", width=460, height=920)
        self.window.visible = True
        self._build()

    def _build(self) -> None:
        ui = self.ui
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label("G1 29DOF 关节映射可视化", height=24)
                ui.Label("拖动滑条可实时修改关节角（单位：rad）", height=18)

                with ui.HStack(height=28, spacing=8):
                    ui.Button("Reset 默认姿态", clicked_fn=self._reset_default)
                    ui.Button("Zero 全部置零", clicked_fn=self._zero_all)
                    ui.Button("Print 当前姿态", clicked_fn=self._print_pose)

                ui.Spacer(height=4)
                with ui.ScrollingFrame(height=820):
                    with ui.VStack(spacing=4):
                        for idx, name in enumerate(self.sdk_joint_names):
                            with ui.HStack(height=24):
                                ui.Label(f"{idx:02d} {name}", width=285)
                                slider = ui.FloatSlider(min=-2.8, max=2.8, step=0.001, width=140)
                                slider.model.set_value(float(self.current_angles[idx]))
                                slider.model.add_value_changed_fn(self._make_slider_cb(idx))
                                self._sliders.append(slider)

    def _make_slider_cb(self, idx: int):
        def _cb(model):
            self.current_angles[idx] = float(model.get_value_as_float())

        return _cb

    def _sync_sliders_from_angles(self) -> None:
        for i, slider in enumerate(self._sliders):
            slider.model.set_value(float(self.current_angles[i]))

    def _reset_default(self) -> None:
        self.current_angles = self.default_angles[:]
        self._sync_sliders_from_angles()
        print("[UI] 已恢复默认姿态。")

    def _zero_all(self) -> None:
        self.current_angles = [0.0] * len(self.current_angles)
        self._sync_sliders_from_angles()
        print("[UI] 已将全部关节置零。")

    def _print_pose(self) -> None:
        print("\n[当前 29 个关节角]")
        for i, n in enumerate(self.sdk_joint_names):
            print(f"  {i:02d} {n:<28} = {self.current_angles[i]: .6f} rad")


def run() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.02)
    sim = SimulationContext(sim_cfg)
    scene_cfg = JointMapperSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    env_ids = torch.tensor([0], dtype=torch.long, device=sim.device)

    sdk_joint_names = list(UNITREE_G1_29DOF_CFG.joint_sdk_names)
    joint_name_to_sim_id: dict[str, int] = {}
    for name in sdk_joint_names:
        joint_ids, _ = robot.find_joints(name)
        if len(joint_ids) == 0:
            raise RuntimeError(f"无法在仿真关节中找到: {name}")
        joint_name_to_sim_id[name] = int(joint_ids[0])

    base_root_state = robot.data.default_root_state.clone()
    base_joint_pos = robot.data.default_joint_pos.clone()
    base_joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    default_angles = [float(base_joint_pos[0, joint_name_to_sim_id[n]].item()) for n in sdk_joint_names]

    print("\n[G1 29DOF joint_sdk_names 映射]")
    for i, n in enumerate(sdk_joint_names):
        print(f"  {i:02d}: {n}")

    ui_state = JointMapperUI(sdk_joint_names=sdk_joint_names, default_angles=default_angles)

    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        joint_pos = base_joint_pos.clone()
        for i, name in enumerate(sdk_joint_names):
            sim_id = joint_name_to_sim_id[name]
            joint_pos[0, sim_id] = ui_state.current_angles[i]

        # 教学演示模式：直接写状态 + 渲染，不做动力学积分。
        robot.write_root_state_to_sim(base_root_state, env_ids=env_ids)
        robot.write_joint_state_to_sim(joint_pos, base_joint_vel, env_ids=env_ids)
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

    print("[INFO] UI joint mapper 已退出。")


if __name__ == "__main__":
    run()
    simulation_app.close()
