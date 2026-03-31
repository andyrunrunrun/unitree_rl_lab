# Repository Guidelines

## Project Structure & Module Organization
Core Python code lives in `source/unitree_rl_lab/unitree_rl_lab/`. Put RL environments under `tasks/` (`locomotion/`, `mimic/`), robot assets under `assets/robots/`, and shared helpers under `utils/`. CLI entrypoints live in `scripts/`: use `scripts/rsl_rl/` for training and inference and `scripts/mimic/` for dataset conversion or replay. Deployment code is separate: shared C++ headers are in `deploy/include/`, and each robot controller has its own `deploy/robots/<robot>/` directory. Treat `deploy/thirdparty/` as vendored code unless you are intentionally upgrading dependencies.

## Build, Test, and Development Commands
Use the repo helper script inside the Isaac Lab conda environment:

- `./unitree_rl_lab.sh -i` installs the package in editable mode and wires shell helpers.
- `./unitree_rl_lab.sh -l` lists registered Unitree tasks.
- `./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity` starts headless training.
- `./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity` runs inference with a trained policy.
- `pre-commit run --all-files` runs Black, isort, Flake8, pyupgrade, codespell, and file hygiene hooks.
- `pyright` performs the configured Python type check pass.
- `cmake .. && make` from `deploy/robots/<robot>/build` builds a robot controller.

## Coding Style & Naming Conventions
Python targets 3.10. Format with Black and isort, and keep lines at 120 characters. Flake8 uses Google-style docstrings and ignores a few Black-conflicting rules; rely on `pre-commit` instead of hand-formatting. Use `snake_case` for modules, functions, and config files such as `velocity_env_cfg.py`; use `PascalCase` for classes. Follow existing task naming patterns like `rsl_rl_ppo_cfg.py` and keep per-robot config beside the robot package it serves.

## Testing Guidelines
There is no dedicated `tests/` tree yet, so contributors should use targeted smoke tests. At minimum, run `pre-commit run --all-files`, `pyright`, `./unitree_rl_lab.sh -l`, and the relevant `train.py` or `play.py` command for the task you changed. For deploy changes, rebuild the affected controller under `deploy/robots/<robot>/build`.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects, sometimes with a lowercase prefix such as `doc:` or `feature:`. Keep commits focused and describe the behavior change, not the implementation detail. Pull requests should state the affected robot or task, list validation commands, link related issues, and include screenshots or logs when changing simulation, deployment, or CLI behavior.

## Configuration Tips
Local asset paths such as `UNITREE_MODEL_DIR` and `UNITREE_ROS_DIR` are configured in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`. Do not commit machine-specific paths, generated logs, datasets, checkpoints, or build outputs.
