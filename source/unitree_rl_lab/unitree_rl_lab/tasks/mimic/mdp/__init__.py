"""Mimic（动作模仿/动作跟踪）任务的 MDP 组件集合。

本目录的代码用于把“参考动作”（motion / demonstration）以 **Command** 的形式注入环境，
并提供围绕该参考动作的：

- **observations**：给策略/价值网络的观测（例如：参考动作在机器人坐标系下的 anchor 误差等）
- **rewards**：跟踪误差的奖励项（位置/姿态/速度的 exp kernel）
- **terminations**：偏差过大或姿态异常时的终止条件
- **events**：域随机化/校准误差建模等（例如：关节零位偏差）

你可以把 Mimic 的训练理解为：
策略输出关节动作 → 机器人在仿真中运动 → 计算“机器人动作”与“参考动作”之间的误差 → 用误差构造奖励并学习。

代码组织上，这里通过 `from isaaclab.envs.mdp import *` 复用 Isaac Lab 通用 MDP 函数，
同时暴露本目录的 Mimic 专用实现，便于在 `tracking_env_cfg.py` 里直接引用 `mdp.xxx`。
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from unitree_rl_lab.tasks.mimic.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
