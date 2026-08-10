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
* [🚀 Installation & Setup](#-installation--setup)
* [💻 CLI Usage (How to Run)](#-cli-usage-how-to-run)

---

## 📌 Project Overview

> **[🚧 Work in Progress]**
---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/KIRCA18/drone-rl.git
cd drone-rl
```

### 2. Install Core Dependencies
Install the required packages, including `typer` for the CLI and `setuptools<82` (required to bypass legacy packaging issues in PyBullet):
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