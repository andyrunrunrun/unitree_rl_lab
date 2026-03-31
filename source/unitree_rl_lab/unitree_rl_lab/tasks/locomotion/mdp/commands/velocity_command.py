from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """速度指令配置（扩展版）。

    该类继承自 IsaacLab 的 `UniformVelocityCommandCfg`，用于描述“目标速度指令”的采样方式与范围。
    在本仓库里，它额外增加了 `limit_ranges` 字段，典型用法是：

    - **训练**：使用较窄的 `ranges`，让策略先学会稳定、可控的运动。
    - **推理/测试**：将 `ranges` 切换为更宽的 `limit_ranges`，评估策略在更大速度范围下的泛化能力。

    设计动机：
    - 训练时直接采样过大的速度指令，容易导致早期探索不稳定、摔倒频繁、学习变慢；
    - 但最终部署又希望覆盖更宽的速度需求，因此通过“训练范围/测试范围”分离更实用。
    """

    # 更宽的速度采样范围（通常在 play/inference 配置中替换掉 `ranges` 使用）。
    # 类型沿用基类的 `Ranges` 数据结构，保持与 IsaacLab command 生成逻辑兼容。
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
