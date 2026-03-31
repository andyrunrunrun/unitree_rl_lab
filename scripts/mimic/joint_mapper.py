"""G1 29DOF 关节映射可视化小工具。

用途：
1) 在 Isaac Sim 中创建一台 G1 人形机器人；
2) 按 `UNITREE_G1_29DOF_CFG.joint_sdk_names` 打印 29 个关节索引；
3) 支持在终端实时输入命令，修改任意关节角度（弧度），直观看“名字 <-> 关节动作”的对应关系。

运行示例：
    python scripts/mimic/joint_mapper.py
    python scripts/mimic/joint_mapper.py --headless   # 无界面模式（一般仅调试用）

交互命令：
    help
    list
    show
    set <index|joint_name> <rad>
    add <index|joint_name> <delta_rad>
    reset
    zero
    quit
"""

from __future__ import annotations

import argparse
import queue
import shlex
import threading
from dataclasses import dataclass

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Interactive G1 joint angle mapper.")
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
    """简单场景：地面 + 灯光 + 一台 G1。"""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@dataclass
class JointCommand:
    op: str
    key: str | None = None
    value: float | None = None


def _print_help() -> None:
    print(
        "\n[命令说明]\n"
        "  help                              显示帮助\n"
        "  list                              打印 [index -> joint_name] 映射\n"
        "  show                              打印当前 29 个 sdk 关节角（rad）\n"
        "  set <idx|name> <rad>             设置关节角\n"
        "  add <idx|name> <delta_rad>       在当前角度上增量修改\n"
        "  reset                             恢复为模型默认姿态\n"
        "  zero                              全部关节置 0\n"
        "  quit                              退出脚本\n"
    )


def _resolve_name(key: str, joint_names: list[str]) -> str | None:
    """把 index/name 统一解析为 joint_name。"""
    if key.isdigit():
        idx = int(key)
        if 0 <= idx < len(joint_names):
            return joint_names[idx]
        return None
    if key in joint_names:
        return key
    return None


def _stdin_worker(cmd_q: queue.Queue[JointCommand], stop_flag: threading.Event) -> None:
    """后台读取 stdin，避免阻塞渲染循环。"""
    _print_help()
    while not stop_flag.is_set():
        try:
            raw = input("joint-mapper> ").strip()
        except EOFError:
            cmd_q.put(JointCommand(op="quit"))
            return
        if not raw:
            continue
        tokens = shlex.split(raw)
        op = tokens[0].lower()
        if op in ("help", "list", "show", "reset", "zero", "quit"):
            cmd_q.put(JointCommand(op=op))
            continue
        if op in ("set", "add") and len(tokens) == 3:
            try:
                val = float(tokens[2])
            except ValueError:
                print("[ERROR] 数值格式错误，应为浮点数（弧度）。")
                continue
            cmd_q.put(JointCommand(op=op, key=tokens[1], value=val))
            continue
        print("[ERROR] 命令无效，输入 help 查看用法。")


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
    current_sdk_angles = torch.tensor(
        [float(base_joint_pos[0, joint_name_to_sim_id[n]].item()) for n in sdk_joint_names], device=sim.device
    )

    print("\n[G1 29DOF joint_sdk_names 映射]")
    for i, n in enumerate(sdk_joint_names):
        print(f"  {i:02d}: {n}")

    cmd_q: queue.Queue[JointCommand] = queue.Queue()
    stop_flag = threading.Event()
    thread = threading.Thread(target=_stdin_worker, args=(cmd_q, stop_flag), daemon=True)
    thread.start()

    should_quit = False
    sim_dt = sim.get_physics_dt()

    while simulation_app.is_running() and not should_quit:
        while True:
            try:
                cmd = cmd_q.get_nowait()
            except queue.Empty:
                break

            if cmd.op == "help":
                _print_help()
            elif cmd.op == "list":
                for i, n in enumerate(sdk_joint_names):
                    print(f"  {i:02d}: {n}")
            elif cmd.op == "show":
                for i, n in enumerate(sdk_joint_names):
                    print(f"  {i:02d} {n:<28} = {current_sdk_angles[i].item(): .6f} rad")
            elif cmd.op in ("set", "add"):
                assert cmd.key is not None and cmd.value is not None
                name = _resolve_name(cmd.key, sdk_joint_names)
                if name is None:
                    print(f"[ERROR] 无效关节标识: {cmd.key}")
                    continue
                idx = sdk_joint_names.index(name)
                if cmd.op == "set":
                    current_sdk_angles[idx] = cmd.value
                else:
                    current_sdk_angles[idx] += cmd.value
                print(f"[OK] {name} -> {current_sdk_angles[idx].item():.6f} rad")
            elif cmd.op == "reset":
                for i, n in enumerate(sdk_joint_names):
                    current_sdk_angles[i] = base_joint_pos[0, joint_name_to_sim_id[n]]
                print("[OK] 已恢复为默认姿态。")
            elif cmd.op == "zero":
                current_sdk_angles.zero_()
                print("[OK] 已将 29 个 sdk 关节置零。")
            elif cmd.op == "quit":
                should_quit = True
                break

        joint_pos = base_joint_pos.clone()
        for i, name in enumerate(sdk_joint_names):
            sim_id = joint_name_to_sim_id[name]
            joint_pos[0, sim_id] = current_sdk_angles[i]

        # 为了做“姿态映射演示”，这里每帧把状态直接写回并仅渲染，不执行动力学步进。
        robot.write_root_state_to_sim(base_root_state, env_ids=env_ids)
        robot.write_joint_state_to_sim(joint_pos, base_joint_vel, env_ids=env_ids)
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

    stop_flag.set()
    print("[INFO] joint mapper 已退出。")


if __name__ == "__main__":
    run()
    simulation_app.close()
