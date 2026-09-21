import copy
import json
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(help="Run Drone-RL experiments with Ray RLlib.")

ENV_NAME = "DroneHover-v0"
DEFAULT_RESULTS_DIR = Path("experiments/results")


class Algorithm(str):
    ppo = "PPO"
    sac = "SAC"
    td3 = "TD3"
    ddpg = "DDPG"


def load_experiment_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_nested(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    current = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def apply_sweep(config: dict[str, Any], sweep: dict[str, list[Any]]) -> dict[str, Any]:
    from ray import tune

    expanded = copy.deepcopy(config)
    for dotted_path, values in sweep.items():
        set_nested(expanded, dotted_path, tune.grid_search(values))
    return expanded


def short_trial_dirname(trial) -> str:
    return f"{trial.trainable_name}_{trial.trial_id}"


def get_algorithm_config_class(algo_name: str):
    normalized = algo_name.upper()
    if normalized == Algorithm.ppo:
        from ray.rllib.algorithms.ppo import PPOConfig

        return PPOConfig
    if normalized == Algorithm.sac:
        from ray.rllib.algorithms.sac import SACConfig

        return SACConfig
    if normalized in (Algorithm.td3, Algorithm.ddpg):
        raise RuntimeError(
            f"{normalized} isn't available in this environment: it was removed from "
            "ray.rllib.algorithms upstream and now only ships via rllib_contrib, which "
            "requires a separate, older Python/Ray stack. Use the dedicated environment "
            "in rllib_contrib_env/ instead, e.g.:\n\n"
            "    python3.10 -m venv .venv-contrib\n"
            "    source .venv-contrib/bin/activate\n"
            "    pip install -r rllib_contrib_env/requirements.txt\n"
            "    pip install --no-deps -e \"git+https://github.com/utiasDSL/gym-pybullet-drones.git"
            "@df831ee8f6fd9cd823f3ecdfa1474a4ddcd40771#egg=gym_pybullet_drones\"\n"
            "    # rllib_contrib_env/vendor/ already ships the TD3/DDPG source, no build step needed\n"
            "    cd rllib_contrib_env && python rllib_train_contrib.py train "
            f"--config ../rllib_configs/{normalized.lower()}_hover.json\n\n"
            "See rllib_contrib_env/README.md for details."
        )
    raise typer.BadParameter(f"Unsupported algorithm: {algo_name}")


def make_drone_env(env_config: dict[str, Any]):
    import gymnasium as gym
    import numpy as np
    from gymnasium import ObservationWrapper
    from gymnasium.spaces import Box
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.enums import ActionType, ObservationType

    obs_type = ObservationType(env_config.get("obs", "kin"))
    act_type = ActionType(env_config.get("act", "one_d_rpm"))
    scenario = env_config.get("scenario", "baseline")
    observation_noise_std = float(env_config.get("observation_noise_std", 0.0))
    initial_position_range = float(env_config.get("initial_position_range", 0.0))
    initial_z_range = float(env_config.get("initial_z_range", 0.0))
    reward_config = copy.deepcopy(env_config.get("reward", {}))

    initial_xyzs = None
    if scenario == "offset_start":
        initial_xyzs = np.array([[0.15, -0.15, 1.0]], dtype=np.float32)
    elif scenario == "random_start" and initial_position_range <= 0:
        initial_position_range = 0.25

    env = HoverAviary(
        gui=bool(env_config.get("gui", False)),
        obs=obs_type,
        act=act_type,
        initial_xyzs=initial_xyzs,
    )

    class RandomizedStartWrapper(gym.Wrapper):
        def __init__(self, wrapped_env, xy_range: float, z_range: float):
            super().__init__(wrapped_env)
            self.xy_range = xy_range
            self.z_range = z_range

        def reset(self, **kwargs):
            base_xyz = np.asarray(getattr(self.unwrapped, "TARGET_POS", [0.0, 0.0, 1.0]), dtype=np.float32)
            offset = np.zeros(3, dtype=np.float32)
            offset[:2] = self.np_random.uniform(-self.xy_range, self.xy_range, size=2)
            if self.z_range > 0:
                offset[2] = self.np_random.uniform(-self.z_range, self.z_range)
            self.unwrapped.INIT_XYZS = (base_xyz + offset).reshape(1, 3)
            return self.env.reset(**kwargs)

    class RewardShapingWrapper(gym.Wrapper):
        DEFAULTS = {
            "position": {
                "original_weight": 1.0,
                "position_weight": 1.0,
            },
            "stable_hover": {
                "original_weight": 1.0,
                "velocity_weight": 0.20,
                "attitude_weight": 0.15,
                "angular_weight": 0.05,
            },
            "energy_aware": {
                "original_weight": 1.0,
                "velocity_weight": 0.20,
                "attitude_weight": 0.15,
                "angular_weight": 0.05,
                "energy_weight": 0.03,
            },
            "smooth_control": {
                "original_weight": 1.0,
                "velocity_weight": 0.20,
                "attitude_weight": 0.15,
                "angular_weight": 0.05,
                "energy_weight": 0.02,
                "smooth_weight": 0.05,
            },
        }

        def __init__(self, wrapped_env, config: dict[str, Any]):
            super().__init__(wrapped_env)
            self.mode = str(config.get("mode", "baseline"))
            weights = copy.deepcopy(self.DEFAULTS.get(self.mode, {}))
            weights.update(config.get("weights", {}))
            self.weights = weights
            self.alive_bonus = float(config.get("alive_bonus", weights.get("alive_bonus", 0.0)))
            self.target_pos = np.asarray(
                config.get("target_pos", getattr(self.unwrapped, "TARGET_POS", [0.0, 0.0, 1.0])),
                dtype=np.float32,
            )
            self._previous_action = None

        def reset(self, **kwargs):
            self._previous_action = None
            return self.env.reset(**kwargs)

        def _normalized_action(self, action):
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
            high = np.where(np.isfinite(high) & (np.abs(high) > 1e-6), np.abs(high), 1.0)
            if high.size == action.size:
                return action / high
            return action

        def _components(self, action):
            state = np.asarray(self.unwrapped._getDroneStateVector(0), dtype=np.float32)
            position = state[0:3]
            roll_pitch = state[7:9]
            linear_velocity = state[10:13]
            angular_velocity = state[13:16]
            normalized_action = self._normalized_action(action)
            if self._previous_action is None:
                action_delta = np.zeros_like(normalized_action)
            else:
                action_delta = normalized_action - self._previous_action

            return {
                "reward_position_error": float(np.linalg.norm(self.target_pos - position)),
                "reward_velocity_error": float(np.linalg.norm(linear_velocity)),
                "reward_attitude_error": float(np.linalg.norm(roll_pitch)),
                "reward_angular_error": float(np.linalg.norm(angular_velocity)),
                "reward_energy_error": float(np.linalg.norm(normalized_action)),
                "reward_smoothness_error": float(np.linalg.norm(action_delta)),
            }

        def step(self, action):
            observation, original_reward, terminated, truncated, info = self.env.step(action)
            components = self._components(action)
            reward = float(self.weights.get("original_weight", 1.0)) * float(original_reward)
            reward += self.alive_bonus
            reward -= float(self.weights.get("position_weight", 0.0)) * components["reward_position_error"]
            reward -= float(self.weights.get("velocity_weight", 0.0)) * components["reward_velocity_error"]
            reward -= float(self.weights.get("attitude_weight", 0.0)) * components["reward_attitude_error"]
            reward -= float(self.weights.get("angular_weight", 0.0)) * components["reward_angular_error"]
            reward -= float(self.weights.get("energy_weight", 0.0)) * components["reward_energy_error"]
            reward -= float(self.weights.get("smooth_weight", 0.0)) * components["reward_smoothness_error"]

            info = dict(info)
            info.update(components)
            info["reward_original"] = float(original_reward)
            info["reward_shaped"] = float(reward)
            info["reward_mode"] = self.mode
            self._previous_action = self._normalized_action(action)
            return observation, reward, terminated, truncated, info

    class RllibDroneWrapper(gym.Wrapper):
        def __init__(self, wrapped_env):
            super().__init__(wrapped_env)
            self._action_shape = wrapped_env.action_space.shape
            self.observation_space = Box(
                low=np.asarray(wrapped_env.observation_space.low, dtype=np.float32).reshape(-1),
                high=np.asarray(wrapped_env.observation_space.high, dtype=np.float32).reshape(-1),
                dtype=np.float32,
            )
            self.action_space = Box(
                low=np.asarray(wrapped_env.action_space.low, dtype=np.float32).reshape(-1),
                high=np.asarray(wrapped_env.action_space.high, dtype=np.float32).reshape(-1),
                dtype=np.float32,
            )

        def _flatten_observation(self, observation):
            return np.asarray(observation, dtype=np.float32).reshape(-1)

        def reset(self, **kwargs):
            observation, info = self.env.reset(**kwargs)
            return self._flatten_observation(observation), info

        def step(self, action):
            env_action = np.asarray(action, dtype=np.float32).reshape(self._action_shape)
            observation, reward, terminated, truncated, info = self.env.step(env_action)
            return self._flatten_observation(observation), reward, terminated, truncated, info

    class NoisyObservationWrapper(ObservationWrapper):
        def __init__(self, wrapped_env, noise_std: float):
            super().__init__(wrapped_env)
            self.noise_std = noise_std
            self.observation_space = Box(
                low=wrapped_env.observation_space.low,
                high=wrapped_env.observation_space.high,
                shape=wrapped_env.observation_space.shape,
                dtype=np.float32,
            )

        def observation(self, observation):
            noise = self.np_random.normal(0.0, self.noise_std, size=observation.shape)
            return (observation + noise).astype(np.float32)

    if initial_position_range > 0 or initial_z_range > 0:
        env = RandomizedStartWrapper(env, initial_position_range, initial_z_range)

    if reward_config and reward_config.get("mode", "baseline") != "baseline":
        env = RewardShapingWrapper(env, reward_config)

    if observation_noise_std > 0:
        env = NoisyObservationWrapper(env, observation_noise_std)

    return RllibDroneWrapper(env)


def register_drone_env() -> None:
    from ray.tune.registry import register_env

    register_env(ENV_NAME, make_drone_env)


def raise_for_failed_trials(result_grid) -> None:
    errors = getattr(result_grid, "errors", [])
    if errors:
        typer.secho(
            f"WARNING: {len(errors)} RLlib trial(s) failed - see each trial's error.txt. "
            "The completed trials were still saved.",
            fg=typer.colors.YELLOW,
        )


def configure_env_runners(config_obj, experiment_config: dict[str, Any]):
    runners = experiment_config.get("env_runners", {})
    num_runners = int(runners.get("num_env_runners", runners.get("num_rollout_workers", 0)))
    envs_per_runner = int(runners.get("num_envs_per_env_runner", 1))
    rollout_fragment_length = runners.get("rollout_fragment_length", "auto")

    if hasattr(config_obj, "env_runners"):
        return config_obj.env_runners(
            num_env_runners=num_runners,
            num_envs_per_env_runner=envs_per_runner,
            rollout_fragment_length=rollout_fragment_length,
        )

    return config_obj.rollouts(
        num_rollout_workers=num_runners,
        num_envs_per_worker=envs_per_runner,
        rollout_fragment_length=rollout_fragment_length,
    )


def build_rllib_config(experiment_config: dict[str, Any], seed: int | None):
    algo_name = experiment_config["algorithm"].upper()
    config_class = get_algorithm_config_class(algo_name)

    training_config = copy.deepcopy(experiment_config.get("training", {}))
    model_config = copy.deepcopy(experiment_config.get("model", {}))
    evaluation_config = copy.deepcopy(experiment_config.get("evaluation", {}))
    resources_config = copy.deepcopy(experiment_config.get("resources", {}))
    reporting_config = copy.deepcopy(experiment_config.get("reporting", {}))
    exploration_config = copy.deepcopy(experiment_config.get("exploration", {}))

    if model_config:
        training_config["model"] = model_config

    config_obj = config_class()

    if experiment_config.get("api_stack"):
        config_obj = config_obj.api_stack(**experiment_config["api_stack"])

    config_obj = (
        config_obj.environment(env=ENV_NAME, env_config=experiment_config.get("env_config", {}))
        .framework(experiment_config.get("framework", "torch"))
        .training(**training_config)
    )

    if seed is not None:
        config_obj = config_obj.debugging(seed=seed)

    config_obj = configure_env_runners(config_obj, experiment_config)

    if resources_config:
        config_obj = config_obj.resources(**resources_config)

    if evaluation_config:
        config_obj = config_obj.evaluation(**evaluation_config)

    if reporting_config:
        config_obj = config_obj.reporting(**reporting_config)

    if exploration_config:
        config_obj = config_obj.exploration(**exploration_config)

    return config_obj


def run_tuner(
    experiment_config: dict[str, Any],
    *,
    steps: int | None,
    name: str | None,
    results_dir: Path,
    samples: int,
    seed: int | None,
    use_sweep: bool,
):
    import ray
    from ray import tune

    try:
        RunConfig = tune.RunConfig
        CheckpointConfig = tune.CheckpointConfig
    except AttributeError:
        from ray.air import CheckpointConfig, RunConfig

    register_drone_env()

    stop_steps = steps or int(experiment_config.get("stop", {}).get("timesteps_total", 100_000))
    stop_metric = experiment_config.get("stop_metric", "num_env_steps_sampled_lifetime")
    experiment_name = name or experiment_config.get("name", "drone_rllib_experiment")
    config_for_build = copy.deepcopy(experiment_config)
    if use_sweep and experiment_config.get("sweep"):
        config_for_build = apply_sweep(config_for_build, experiment_config["sweep"])

    config_obj = build_rllib_config(config_for_build, seed)
    param_space = config_obj.to_dict()

    seeds = experiment_config.get("seeds")
    if seed is None and seeds:
        param_space["seed"] = tune.grid_search(list(seeds))

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    try:
        tuner = tune.Tuner(
            experiment_config["algorithm"].upper(),
            param_space=param_space,
            tune_config=tune.TuneConfig(
                num_samples=samples,
                max_concurrent_trials=max_concurrent_trials,
                trial_dirname_creator=short_trial_dirname,
            ),
            run_config=RunConfig(
                name=experiment_name,
                storage_path=str(results_dir.resolve()),
                stop={stop_metric: stop_steps},
                checkpoint_config=CheckpointConfig(
                    checkpoint_frequency=int(experiment_config.get("checkpoint_frequency", 10)),
                    checkpoint_at_end=True,
                ),
            ),
        )
        result_grid = tuner.fit()
        raise_for_failed_trials(result_grid)
        return result_grid
    finally:
        ray.shutdown()


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", help="Path to an RLlib experiment JSON config."),
    steps: int | None = typer.Option(None, "--steps", "-s", help="Override total timesteps."),
    name: str | None = typer.Option(None, "--name", "-n", help="Override experiment name."),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir", help="Where Ray Tune writes results."),
    samples: int = typer.Option(1, "--samples", help="Number of Tune samples for stochastic/randomized sweeps."),
    seed: int | None = typer.Option(None, "--seed", help="Pin ONE seed (disables the config's multi-seed grid)."),
    seeds: str | None = typer.Option(None, "--seeds", help="Comma-separated seed list, overrides the config's 'seeds' (e.g. --seeds 0,1,2)."),
    use_sweep: bool = typer.Option(True, "--sweep/--no-sweep", help="Enable or disable grid-search sweep values from the config."),
):
    """Train one RLlib experiment from a JSON config."""
    experiment_config = load_experiment_config(config)
    if seeds is not None:
        experiment_config["seeds"] = [int(s) for s in seeds.split(",") if s.strip() != ""]
    run_tuner(
        experiment_config,
        steps=steps,
        name=name,
        results_dir=results_dir,
        samples=samples,
        seed=seed,
        use_sweep=use_sweep,
    )
    typer.secho("RLlib training finished.", fg=typer.colors.GREEN)


def _latest_checkpoint(trial_dir: Path) -> Path | None:
    ckpts = sorted(trial_dir.glob("checkpoint_*"))
    return ckpts[-1] if ckpts else None


def _truncate_logs_to_iter(trial_dir: Path, keep_through_iter: int) -> None:
    rj = trial_dir / "result.json"
    if rj.exists():
        kept = []
        for line in rj.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                it = json.loads(line).get("training_iteration", 0)
            except json.JSONDecodeError:
                continue
            if it is None or it <= keep_through_iter:
                kept.append(line)
        rj.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    pc = trial_dir / "progress.csv"
    if pc.exists():
        rows = pc.read_text(encoding="utf-8").splitlines()
        if rows:
            header = rows[0].split(",")
            try:
                ci = header.index("training_iteration")
            except ValueError:
                ci = None
            kept = [rows[0]]
            for r in rows[1:]:
                if ci is None:
                    kept.append(r)
                    continue
                cells = r.split(",")
                try:
                    if int(float(cells[ci])) <= keep_through_iter:
                        kept.append(r)
                except (ValueError, IndexError):
                    kept.append(r)
            pc.write_text("\n".join(kept) + "\n", encoding="utf-8")


def _resume_one_trial(trial_dir: Path, stop_metric: str, stop_steps: int, checkpoint_frequency: int) -> None:
    from ray.rllib.algorithms.algorithm import Algorithm

    ckpt = _latest_checkpoint(trial_dir)
    if ckpt is None:
        typer.secho(f"  {trial_dir.name}: no checkpoint_* to resume from - skipping", fg=typer.colors.YELLOW)
        return

    algo = Algorithm.from_checkpoint(str(ckpt))
    result_file = trial_dir / "result.json"
    resume_iter = int(algo.iteration)
    _truncate_logs_to_iter(trial_dir, resume_iter)

    def _steps(res: dict) -> int:
        return int(res.get(stop_metric) or res.get("timesteps_total") or res.get("num_env_steps_sampled_lifetime") or 0)

    def _save(it: int) -> None:
        target = trial_dir / f"checkpoint_{it:06d}"
        target.mkdir(parents=True, exist_ok=True)
        try:
            algo.save_to_path(str(target))          # new API stack
        except (RuntimeError, AttributeError):
            algo.save(str(target))                  # old API stack (SAC / contrib TD3/DDPG)

    typer.secho(f"  {trial_dir.name}: resuming from {ckpt.name} (iter {resume_iter})", fg=typer.colors.CYAN)
    it = resume_iter
    try:
        while True:
            res = algo.train()
            with result_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(res, default=str) + "\n")
            it = int(res.get("training_iteration", algo.iteration))
            if checkpoint_frequency and it % checkpoint_frequency == 0:
                _save(it)
            if _steps(res) >= stop_steps:
                break
        _save(it)
        typer.secho(f"  {trial_dir.name}: reached {_steps(res)} steps (iter {it})", fg=typer.colors.GREEN)
    finally:
        algo.stop()


def resume_experiment(
    experiment_config: dict[str, Any],
    *,
    results_dir: Path,
    name: str | None,
    steps: int | None,
    include_errored: bool,
    only_trial: str | None,
) -> None:
    import ray
    from ray import tune

    register_drone_env()
    experiment_name = name or experiment_config.get("name", "drone_rllib_experiment")
    experiment_dir = (results_dir / experiment_name).resolve()
    if not experiment_dir.exists():
        raise typer.BadParameter(f"No experiment to resume at {experiment_dir}")

    stop_metric = experiment_config.get("stop_metric", "num_env_steps_sampled_lifetime")
    stop_steps = steps or int(experiment_config.get("stop", {}).get("timesteps_total", 100_000))
    checkpoint_frequency = int(experiment_config.get("checkpoint_frequency", 10))

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    try:
        if only_trial:
            matches = sorted(d for d in experiment_dir.iterdir() if d.is_dir() and only_trial in d.name)
            if not matches:
                raise typer.BadParameter(f"No trial dir under {experiment_dir} contains {only_trial!r}")
            typer.secho(f"Resuming {len(matches)} trial(s) matching {only_trial!r}:", fg=typer.colors.GREEN)
            for trial_dir in matches:
                _resume_one_trial(trial_dir, stop_metric, stop_steps, checkpoint_frequency)
            return

        # whole experiment
        tuner = tune.Tuner.restore(
            str(experiment_dir),
            trainable=experiment_config["algorithm"].upper(),
            resume_unfinished=True,
            resume_errored=include_errored,
        )
        result_grid = tuner.fit()
        raise_for_failed_trials(result_grid)
    finally:
        ray.shutdown()


@app.command()
def resume(
    config: Path = typer.Option(..., "--config", "-c", help="The JSON config the experiment was launched with."),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir", help="Where the experiment dir lives."),
    name: str | None = typer.Option(None, "--name", "-n", help="Experiment name override (defaults to the config's 'name')."),
    steps: int | None = typer.Option(None, "--steps", "-s", help="Target total timesteps (defaults to the config's stop)."),
    errored: bool = typer.Option(False, "--errored", help="Also resume ERRORED trials from their last checkpoint (default: only unfinished ones)."),
    trial: str | None = typer.Option(None, "--trial", help="Resume ONLY the trial(s) whose folder name contains this text, e.g. 'gamma=0.9900,seed=0'."),
):
    """Continue an interrupted run: all unfinished trials, or just one with --trial."""
    experiment_config = load_experiment_config(config)
    resume_experiment(
        experiment_config,
        results_dir=results_dir,
        name=name,
        steps=steps,
        include_errored=errored,
        only_trial=trial,
    )
    typer.secho("RLlib resume finished.", fg=typer.colors.GREEN)


@app.command()
def batch(
    config_dir: Path = typer.Option(Path("rllib_configs"), "--config-dir", help="Directory with RLlib JSON configs."),
    steps: int | None = typer.Option(None, "--steps", "-s", help="Override total timesteps for every config."),
    seeds: str = typer.Option("0,1,2", "--seeds", help="Comma-separated seeds to run for every config."),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir", help="Where Ray Tune writes results."),
):
    """Run all configs in a directory across multiple seeds."""
    config_paths = sorted(config_dir.glob("*.json"))
    if not config_paths:
        raise typer.BadParameter(f"No JSON configs found in {config_dir}")

    parsed_seeds = [int(seed.strip()) for seed in seeds.split(",") if seed.strip()]
    for config_path in config_paths:
        base_config = load_experiment_config(config_path)
        for seed in parsed_seeds:
            run_name = f"{base_config.get('name', config_path.stem)}_seed_{seed}"
            typer.echo(f"Starting {run_name}")
            run_tuner(
                base_config,
                steps=steps,
                name=run_name,
                results_dir=results_dir,
                samples=1,
                seed=seed,
                use_sweep=True,
            )
    typer.secho("RLlib comparison finished.", fg=typer.colors.GREEN)


@app.command("show-config")
def show_config(config: Path = typer.Argument(..., help="Path to an RLlib experiment JSON config.")):
    """Print a normalized config before training."""
    experiment_config = load_experiment_config(config)
    typer.echo(json.dumps(experiment_config, indent=2))


if __name__ == "__main__":
    app()
