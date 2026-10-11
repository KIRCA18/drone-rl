# Drone-RL

A comparative study of PPO, SAC, TD3, and DDPG for simulated drone hovering,
exploring training stability, policy collapse, and reward shaping with Ray RLlib
and PyBullet.

Developed for the Agent-Based Systems course at FINKI by
[Veronika Ilioska](https://github.com/veronika-ilioska) and
[Kiril Veljanoski](https://github.com/KIRCA18).

## Overview

The agent learns to take off and hover at a fixed target in the
`gym-pybullet-drones` HoverAviary environment. It receives kinematic observations
and controls the drone through a single collective motor command (`one_d_rpm`).

The project includes:

- PPO and SAC experiments in the main RLlib environment.
- TD3 and DDPG experiments in a separate compatibility environment.
- Configuration files for multiple seeds, hyperparameter sweeps, and reward shaping.
- Checkpoint saving, training resumption, and policy visualization.
- Result summaries and learning-curve plots.

## Setup

Clone the repository, then create and activate a virtual environment:

```powershell
git clone https://github.com/KIRCA18/drone-rl.git
cd drone-rl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Training

RLlib experiments are defined in [rllib_configs](rllib_configs). Each file
specifies the algorithm, environment, training parameters, evaluation settings,
and any seed or hyperparameter sweep.

Run one short PPO trial:

```powershell
python main.py rllib train --config rllib_configs/ppo_hover.json --steps 4000 --seed 0 --no-sweep
```

Train SAC using the budget and seeds in its configuration:

```powershell
python main.py rllib train --config rllib_configs/sac_hover.json
```

Limit the number of concurrent trials when running multiple seeds or a sweep:

```powershell
python main.py rllib train --config rllib_configs/sac_hover_reward_modes.json --max-concurrent-trials 2
```

Resume an interrupted experiment using its original configuration:

```powershell
python main.py rllib resume --config rllib_configs/sac_hover.json
```

Use `python main.py rllib --help` to see the available commands. The `batch`
command runs every configuration in a selected directory; use a directory of
PPO/SAC configurations with the main environment and the separate runner below
for TD3/DDPG.

## Reward Shaping

Baseline configurations use the original HoverAviary reward. The
`env_config.reward` setting supports additional penalties:

| Mode | Additional penalties |
| --- | --- |
| `baseline` | None |
| `position` | Distance to the target |
| `stable_hover` | Linear velocity, roll/pitch, and angular velocity |
| `energy_aware` | Stability terms and action magnitude |
| `smooth_control` | Stability terms, action magnitude, and action changes |

The action-magnitude term is a control-effort proxy, not a measurement of battery
consumption. Returns from different reward functions should be evaluated on a
common reward before being compared directly.

Run all four shaped SAC reward modes:

```powershell
python main.py rllib train --config rllib_configs/sac_hover_reward_modes.json
```

Train with randomized starting positions and a smooth-control reward:

```powershell
python main.py rllib train --config rllib_configs/sac_hover_smooth_random_start.json
```

TD3/DDPG configurations also support potential-based progress shaping through
`env_config.reward_shaping`. The shaped-v2 variants add action-magnitude and
vertical-velocity penalties; shaped-v3 gradually reduces those penalties during
training.

## Results and Evaluation

Summarize saved RLlib results from the main environment:

```powershell
python main.py rllib-results summarize
```

The command writes timestamped outputs to:

- `experiments/results/`: per-trial CSV summaries alongside training logs and checkpoints.
- `experiments/plots/`: learning-curve plots.
- `reports/`: Markdown comparison reports.

The summary plot displays the top 10 trials by default. The CSV and Markdown
report include all trials with usable return data. Evaluation returns are
preferred; training returns are used when no evaluation measurements are logged.

## Visualizing a Policy

Replace `PATH_TO_CHECKPOINT` with a saved Ray Tune checkpoint directory:

```powershell
python main.py rllib visualize --checkpoint PATH_TO_CHECKPOINT --config rllib_configs/sac_hover.json --episodes 1 --no-explore
```

The visualizer opens the PyBullet GUI. Use the configuration corresponding to
the saved policy.

## TD3 and DDPG

These algorithms use the vendored `rllib_contrib` implementation and a separate
Python 3.10 environment with Ray 2.5.1. The main environment uses Ray 2.58.0.
See [third-party attribution](rllib_contrib_env/README.md) for the source and
license of the vendored code.

Create the compatibility environment from the repository root:

```powershell
py -3.10 -m venv rllib_contrib_env/.venv
.\rllib_contrib_env\.venv\Scripts\Activate.ps1
python -m pip install -r rllib_contrib_env/requirements.txt
python -m pip install --no-deps "git+https://github.com/utiasDSL/gym-pybullet-drones.git@df831ee8f6fd9cd823f3ecdfa1474a4ddcd40771#egg=gym_pybullet_drones"
```

On Linux or macOS, use `python3.10` to create the environment and
`source rllib_contrib_env/.venv/bin/activate` to activate it. The simulator commit
above differs from the main requirements file; align simulator versions when
setting up a controlled comparison.

If the simulator reports `ModuleNotFoundError: No module named 'pkg_resources'`,
install `setuptools<81` in this environment.

Run experiments from the contrib directory:

```powershell
cd rllib_contrib_env
python rllib_train_contrib.py train --config ../rllib_configs/td3_hover.json --steps 4000 --no-sweep --seed 0
python rllib_train_contrib.py train --config ../rllib_configs/td3_hover_shaped_v2.json
python rllib_train_contrib.py train --config ../rllib_configs/ddpg_hover_shaped_v2.json
```

Visualize a saved TD3 policy:

```powershell
python rllib_train_contrib.py visualize --checkpoint PATH_TO_CHECKPOINT --config ../rllib_configs/td3_hover_shaped_v2.json
```

Results are saved in the shared `experiments/results/` directory. Switch back to
the main environment and repository root to run the summary command.

## Stable-Baselines3 Interface

The project also provides a simpler training and rendering interface for PPO,
SAC, and TD3:

```powershell
python main.py train --algo SAC --steps 500000 --output models/sac_hover_v1
python main.py render --algo SAC --model models/sac_hover_v1
python main.py train-and-render --algo TD3 --steps 100000
```

This interface saves Stable-Baselines3 models separately from RLlib checkpoints.
Use `python main.py --help` for all available options.
