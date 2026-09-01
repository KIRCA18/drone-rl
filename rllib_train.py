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

    initial_xyzs = None
    if scenario == "offset_start":
        initial_xyzs = np.array([[0.15, -0.15, 1.0]], dtype=np.float32)

    env = HoverAviary(
        gui=bool(env_config.get("gui", False)),
        obs=obs_type,
        act=act_type,
        initial_xyzs=initial_xyzs,
    )

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
            tune_config=tune.TuneConfig(num_samples=samples),
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
