# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""使用 RSL-RL 训练（训练入口脚本）。

该脚本把“训练需要的配置/环境/Runner/日志”串起来，主要流程如下：

1) 通过 Gymnasium/Isaac Lab 的注册表收集可用的 Unitree 任务 id（用于命令行 `--task` 的候选列表）；
2) 解析命令行参数（包含 RSL-RL 超参、Isaac Sim/Kit 启动参数，以及是否录制视频/分布式等）；
3) 启动 Isaac Sim/Kit（底层 Kit/渲染/物理运行时）；
4) 通过 Hydra 根据 `env_cfg` / `agent_cfg` 创建环境并构建训练 Runner；
5) 可选：视频录制、分布式训练、断点续训/蒸馏；
6) 调用 `runner.learn(...)` 开始训练，并把 env/agent/部署配置写入日志目录，便于实验复现与排障。
"""

"""启动 Isaac Sim/Kit（在创建仿真/场景之前必须执行）。

说明：
- Isaac Lab 的任务通常依赖 Kit 的渲染/物理/传感器管线；
- 因此在 `gym.make(...)` 创建 env 之前，必须确保应用已启动并完成初始化。
"""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
# 触发导入：把 `unitree_rl_lab` 里所有任务模块加载进来，
# 从而让 gym registry 能识别这些环境注册入口（否则 `--task` 选项可能为空）。
from list_envs import import_packages  # noqa: F401

# 导入完成后把 sys.path 恢复，减少对其它 import 的影响（避免“隐式导入”导致歧义）。
sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        # task_spec.id 就是命令行中 `--task` 选择的任务名字。
        # 这里通过字符串过滤只保留本扩展的 Unitree 任务，排除一些包含 Isaac 的注册条目
        # （避免把上游 Isaac Lab 的条目也混进可选列表）。
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Seed used for the environment (用于控制初始化/域随机化等随机源的可复现性)。",
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# argcomplete 允许命令行对参数进行自动补全（当系统已安装/启用且环境支持时）。
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# 如果开启了 `--video`，则强制打开相机渲染能力。
# 原因：在某些任务/渲染管线里默认不会输出可用于视频录制的 `rgb_array`。
if args_cli.video:
    args_cli.enable_cameras = True

# Hydra 会从 sys.argv 里解析配置相关参数。
# 这里把 argparse 无关的参数替换成 `hydra_args`，避免 argparse 的参数与 Hydra 的参数发生冲突。
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Omniverse/Isaac Sim 应用（底层 Kit/渲染/物理运行时）。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""检查是否满足最小 RSL-RL 版本要求（尤其是分布式训练时）。

理由：
- 分布式训练/Runner 行为对依赖版本更敏感；
- 若版本不足，训练阶段才失败会浪费大量启动与仿真时间；
- 因此尽早给出可执行的安装命令并退出。
"""

import importlib.metadata as metadata
import platform

from packaging import version

# 如果开启了 distributed，脚本会检查 rsl-rl-lib 的版本是否满足最小要求。
# 若不满足，给出安装正确版本的命令并退出，避免训练过程中因版本不兼容导致失败。
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import shutil
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 开启 TF32 通常能提升 GPU 上的矩阵运算吞吐。
# deterministic=False 表示允许非确定性算法以换取性能（通常能加速训练）。
# cudnn.benchmark=False 表示不让 cuDNN 针对当前输入形状做额外的内核搜索开销。
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # 使用 Hydra 根据 task 配置加载 env_cfg / agent_cfg，然后用非 Hydra 的 CLI 参数覆盖其中部分字段。
    #
    # 设计意图：
    # - Hydra 负责“完整配置”的组织与默认值；
    # - CLI 负责“少量覆盖”（例如 num_envs/max_iterations/seed/device/video/distributed）。
    #
    # 这样可以在保持配置可复现的同时，方便快速做实验对比。
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # 2. 设置环境随机种子（seed）
    # 注意：环境在初始化阶段就可能进行随机化（比如域随机化、初始状态扰动等），
    # 所以必须在 `gym.make(...)` 创建环境之前把 seed 写到 env_cfg 里。
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 3. 分布式/多 GPU 训练配置
    if args_cli.distributed:
        # 把环境仿真与策略训练都绑定到当前进程的 GPU（local_rank）。
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # 不同 rank 的采样希望有一定差异，因此给每个进程一个不同 seed（在原 seed 上偏移 rank）。
        # 这能减少多进程采样到高度相同轨迹的概率，让并行数据更有“信息增益”。
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # 4. 组织实验日志目录
    # experiment_name 来自 agent_cfg（配置文件中定义），日志根目录固定为 logs/rsl_rl/...
    # log_root_path：实验的根目录（共享一个 experiment_name 下的多个 run）。
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # log_dir 再加上时间戳（以及可选 run_name），避免覆盖旧实验；同时让你快速按时间定位结果。
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 这里保留一种命名方式：一些外部工具（比如 Ray Tune）可能依赖目录名解析实验名。
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # 5. 创建 Isaac Lab 环境
    # gym.make 会从 gym registry 找到 task 的注册入口，并实例化 env（传入 cfg=env_cfg）。
    # render_mode 用于视频录制：
    # - 启用 video 时返回 rgb_array 给 RecordVideo wrapper 使用；
    # - 禁用 video 时传 None，降低渲染相关开销（更快、更省资源）。
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 6. 如果 env 是 Multi-Agent（DirectMARLEnv），而 rsl-rl 算法要求 single-agent，
    # 则转换为 single-agent 形式以匹配训练接口。
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 7. 断点续训/蒸馏：在开始写新 log_dir 之前先定位 resume checkpoint 路径。
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        # 先算出 resume_path，保证后续 runner.load() 能定位到正确的权重文件。
        # 这里的 resume_path 推导逻辑在 get_checkpoint_path 内部完成，
        # 会根据 log_root_path、load_run、load_checkpoint 等配置组合出最终文件路径。
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # 8. 可选：视频录制 wrapper
    # 通过 step_trigger 在指定 step 间隔触发录制，把视频保存到 log_dir/videos/train。
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            # step_trigger(step) 返回 True 时开始录制一段视频。
            # 使用模运算：每隔 video_interval steps 触发一次录制。
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            # RecordVideo wrapper 自带日志；训练日志也会记录信息，因此关闭 wrapper 日志减少噪音。
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 9. 为 rsl-rl 包装环境（VecEnv 形式）
    # clip_actions：对动作进行裁剪，避免策略输出超出动作空间边界导致仿真不稳定/数值发散。
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 10. 构建训练 Runner
    # OnPolicyRunner 负责：
    # - 采样交互数据（rollout）
    # - 计算 advantage/returns
    # - 优化策略与价值网络
    # device：Runner 在该设备上执行（单卡时对应同一块 GPU；分布式时由 local_rank 注入）。
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # 把当前 git 仓库状态写入日志（利于实验复现）。
    runner.add_git_repo_to_log(__file__)
    # 11. 加载断点续训 checkpoint（如开启 resume 或算法是 Distillation）。
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # 12. 把训练配置与部署配置导出到日志目录
    # - params/env.yaml、params/agent.yaml：包含 Hydra 展开后的最终参数值，便于精确复现
    # - export_deploy_cfg：导出部署相关配置（如控制器/模型接口等），避免训练与部署配置漂移
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # 复制 env_cfg 对应的 Python 配置源码文件，便于追溯当次训练使用的“实现来源”（而不仅是最终参数）。
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # 13. 开始训练
    # init_at_random_ep_len=True：随机初始化 episode 长度，有助于减少训练初期的统计偏差。
    # 直观理解：让 rollout 的“回合边界”不会总从完全相同的时间步开始，提升初期训练稳定性。
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()  # pyright: ignore[reportCallIssue]
    # close sim app
    simulation_app.close()
