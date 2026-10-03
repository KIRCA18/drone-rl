<div align="center">
  <h1>🚁 Drone-RL</h1>
  <p><strong>A Comparative Analysis of Continuous Control RL Algorithms for Nano-Quadrotors</strong></p>
  <p><em>Final Project for Agent-Based Systems (FINKI)</em></p>
  <p><strong>Authors:</strong> 
    Veronika Ilioska (<a href="https://github.com/veronika-ilioska">GitHub</a> | <a href="https://www.linkedin.com/in/veronika-ilioska-42655a243/">LinkedIn</a>) &
    Kiril Veljanoski (<a href="https://github.com/KIRCA18">GitHub</a> | <a href="https://www.linkedin.com/in/kiril-veljanoski-1877b9235/">LinkedIn</a>) 
  </p>
</div>

---

## 📋 Table of Contents
* [📌 Project Overview](#-project-overview)
* [🧪 RLlib Experiment Plan](#-rllib-experiment-plan)
* [🚀 Installation & Setup](#-installation--setup)
* [💻 CLI Usage (How to Run)](#-cli-usage-how-to-run)
* [🧠 RLlib Experiments](#-rllib-experiments)
* [🔬 TD3 & DDPG (separate environment)](#-td3--ddpg-separate-environment)

---

## 📌 Project Overview

> **[🚧 Work in Progress]**
---

## 🧪 RLlib Experiment Plan

The comparative experiment design is documented in [RLIB_EXPERIMENT_PLAN.md](RLIB_EXPERIMENT_PLAN.md). It includes algorithm comparisons, scenario/subset comparisons, hyperparameter optimization, neural network architecture combinations, RLlib configuration ideas, and evaluation metrics.

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/KIRCA18/drone-rl.git
cd drone-rl
```

### 2. Install Core Dependencies
Install the required packages, including `typer` for the CLI, Ray RLlib, and PyBullet:
```bash
pip install -r requirements.txt
```


---

## 💻 CLI Usage (How to Run)

This project features a fully automated Command Line Interface (CLI) built with `typer`. You can view the complete help menu at any time by running:
```bash
python main.py --help
```

### 🎯 1. Train a Policy
Trains an agent on your CPU without opening a visualizer. By default, it automatically optimizes parallel environment workers (`n_envs=4` for PPO, `n_envs=1` for SAC/TD3) to prevent CPU bottlenecks.
```bash
# Default: Train PPO for 1,000,000 steps
python main.py train

# Train SAC for 500,000 steps with a custom output name
python main.py train --algo SAC --steps 500000 --output models/sac_hover_v1
```

### 👁️ 2. Render a Saved Model
Loads a saved model from disk and launches the PyBullet 3D visual interface so you can watch the learned policy in action.
```bash
# Render the model you just trained
python main.py render --algo SAC --model models/sac_hover_v1
```

### ⚡ 3. Train and Immediately Render
Runs a training session and immediately launches the PyBullet visualizer upon completion. Perfect for quick testing.
```bash
# Run a quick 100k-step test with TD3
python main.py train-and-render --algo TD3 --steps 100000
```

---

## 🧠 RLlib Experiments

The project includes a Ray RLlib implementation for algorithm comparison, scenario comparison, hyperparameter sweeps, and result analysis.

You can run RLlib through the main CLI:
```bash
python main.py rllib --help
```

### Train one RLlib experiment
```bash
# Quick smoke test
python main.py rllib train --config rllib_configs/ppo_hover.json --steps 1000 --no-sweep

# Longer PPO training run
python main.py rllib train --config rllib_configs/ppo_hover.json --steps 100000
```

### Try shaped hover rewards

The RLlib environment supports optional reward shaping through `env_config.reward`.
Baseline configs keep the original `HoverAviary` reward unchanged; shaped configs
add penalties for unstable flight behavior such as high velocity, roll/pitch,
angular velocity, excessive motor effort, or sudden action changes.

```bash
# PPO with velocity + attitude + angular-rate penalties
python main.py rllib train --config rllib_configs/ppo_hover_stable_reward.json --steps 100000

# SAC with the same stable-hover reward
python main.py rllib train --config rllib_configs/sac_hover_stable_reward.json --steps 100000

# SAC with stable-hover + energy/smoothness penalties and randomized starts
python main.py rllib train --config rllib_configs/sac_hover_smooth_random_start.json --steps 100000

# Sweep several reward modes in one Ray Tune run
python main.py rllib train --config rllib_configs/sac_hover_reward_modes.json --steps 100000
```

Available reward modes:

* `baseline`: original PyBullet hover reward.
* `position`: original reward plus a distance-to-target penalty.
* `stable_hover`: original reward plus velocity, roll/pitch, and angular-rate penalties.
* `energy_aware`: `stable_hover` plus motor effort penalty.
* `smooth_control`: `energy_aware` plus action-change penalty.

Randomized starts can be enabled with:

```json
"env_config": {
  "scenario": "random_start",
  "initial_position_range": 0.25,
  "initial_z_range": 0.10
}
```

### Compare all configured algorithms across seeds
```bash
python main.py rllib compare --config-dir rllib_configs --steps 100000 --seeds 0,1,2
```

### Analyze Ray Tune results
```bash
python main.py rllib-results summarize
```

Outputs are written to:

* `experiments/results/`
* `experiments/plots/`
* `reports/algorithm_comparison.md`

### Visualize an RLlib checkpoint

After training, point the visualizer at a Ray Tune checkpoint directory to watch
the policy in the PyBullet GUI:

```bash
python main.py rllib visualize \
  --checkpoint experiments/results/ppo_hover/PPO_DroneHover-v0_.../checkpoint_000010 \
  --config rllib_configs/ppo_hover.json
```

Useful options:

```bash
python main.py rllib visualize --help
python main.py rllib visualize --checkpoint PATH --episodes 1 --max-steps 1000 --no-explore
```

---

## 🔬 TD3 & DDPG (separate environment)

Ray removed TD3 and DDPG from `ray.rllib.algorithms` — they only survive as the
archived `rllib_contrib` packages, which are hard-pinned to `ray[rllib]==2.5.x`
and **Python < 3.11**. That cannot share the main virtualenv (`ray[rllib]==2.58`,
`gymnasium>=1.0`), so TD3/DDPG run from a second, isolated environment in
`rllib_contrib_env/`. The algorithm source is vendored under
`rllib_contrib_env/vendor/` (see `rllib_contrib_env/README.md` for attribution —
it is copied verbatim from `ray-project/ray` at tag `ray-2.9.0`, Apache-2.0).

### 1. Create the environment (Python 3.10)

```bash
# from the repo root — needs a Python 3.10 interpreter
python3.10 -m venv rllib_contrib_env/.venv          # Windows: py -3.10 -m venv rllib_contrib_env\.venv

# activate it
source rllib_contrib_env/.venv/bin/activate         # Windows: rllib_contrib_env\.venv\Scripts\activate

pip install -r rllib_contrib_env/requirements.txt

# gym-pybullet-drones, pinned to the SAME commit the main env uses, installed
# with --no-deps so pip doesn't fail on the older gymnasium pin:
pip install --no-deps "git+https://github.com/utiasDSL/gym-pybullet-drones.git@df831ee8f6fd9cd823f3ecdfa1474a4ddcd40771#egg=gym_pybullet_drones"
```

If `import gym_pybullet_drones` later fails with
`ModuleNotFoundError: No module named 'pkg_resources'`, run
`pip install "setuptools<81"` in this venv.

### 2. Run the configs

Run from inside `rllib_contrib_env/`, using the shared configs in `../rllib_configs/`:

```bash
cd rllib_contrib_env

# smoke test
python rllib_train_contrib.py train --config ../rllib_configs/td3_hover.json --steps 4000 --no-sweep --seeds 0

# a sweep (3 values x 5 seeds), short budget
python rllib_train_contrib.py train --config ../rllib_configs/td3_hover_gamma.json  --steps 150000
python rllib_train_contrib.py train --config ../rllib_configs/ddpg_hover_gamma.json --steps 150000
```

Results land in `../experiments/results/` — the same tree PPO/SAC use — so
`python main.py rllib-results summarize` (run from the **main** environment)
picks them up alongside everything else.

To visualize a TD3/DDPG checkpoint, run the contrib visualizer from inside the
dedicated environment:

```bash
cd rllib_contrib_env
python rllib_train_contrib.py visualize \
  --checkpoint ../experiments/results/td3_hover/TD3_DroneHover-v0_.../checkpoint_000010 \
  --config ../rllib_configs/td3_hover.json
```
