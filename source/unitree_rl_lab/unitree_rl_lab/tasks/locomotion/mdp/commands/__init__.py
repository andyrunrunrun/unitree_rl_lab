"""Locomotion MDP 指令（commands）子模块入口。

这里的 `commands` 用于定义“任务指令/目标”（例如目标线速度、角速度），
环境会按照配置的采样策略周期性生成 command，策略网络在观测中接收这些指令并尝试跟踪。

注意：
- 该文件通常只负责 re-export（对外导出）配置类，方便上层通过 `mdp.xxx` 直接访问。
- `# noqa` 是为了兼容 lint 规则：该导入是为了导出符号而存在，可能会被误判为未使用。
"""

from .velocity_command import UniformLevelVelocityCommandCfg  # noqa: F401, F403
