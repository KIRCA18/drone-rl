import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(help="Run Drone-RL TD3 / DDPG experiments through rllib_contrib.")

ENV_NAME = "DroneHover-v0"
DEFAULT_RESULTS_DIR = Path("../experiments/results")
STOP_METRIC = "timesteps_total"

SUPPORTED_ALGORITHMS = {"TD3", "DDPG"}
_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
_VENDOR_SRC_PATHS = [str(_VENDOR_DIR / sub) for sub in ("ddpg/src", "td3/src")]
for _path in _VENDOR_SRC_PATHS:
    if _path not in sys.path:
        sys.path.insert(0, _path)

_existing_pythonpath = os.environ.get("PYTHONPATH", "")
_pythonpath_parts = _VENDOR_SRC_PATHS + ([_existing_pythonpath] if _existing_pythonpath else [])
os.environ["PYTHONPATH"] = os.pathsep.join(_pythonpath_parts)


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
    if normalized not in SUPPORTED_ALGORITHMS:
        raise typer.BadParameter(
            f"Unsupported algorithm for rllib_contrib: {algo_name}. "
            f"This script only handles {sorted(SUPPORTED_ALGORITHMS)}. "
            "Use ../rllib_train.py for PPO/SAC."
        )

    try:
        if normalized == "TD3":
            from rllib_td3.td3 import TD3Config

            return TD3Config
        else:
            from rllib_ddpg.ddpg import DDPGConfig

            return DDPGConfig
    except ImportError as exc:
        raise RuntimeError(
            "Could not import rllib_contrib's vendored TD3/DDPG source "
            f"({exc}). Make sure vendor/ddpg/src and vendor/td3/src exist "
            "next to this script (see rllib_contrib_env/README.md — 'git "
            "sparse-checkout' step) and that you're running this inside the "
            "dedicated Python 3.10 virtualenv with "
            "`pip install -r requirements.txt` already done."
        ) from exc


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
            "Completed trials were still saved.",
            fg=typer.colors.YELLOW,
        )


def configure_env_runners(config_obj, experiment_config: dict[str, Any]):
    runners = experiment_config.get("env_runners", {})
    num_runners = int(runners.get("num_env_runners", runners.get("num_rollout_workers", 0)))
    envs_per_runner = int(runners.get("num_envs_per_env_runner", runners.get("num_envs_per_worker", 1)))
    rollout_fragment_length = runners.get("rollout_fragment_length", "auto")
    obs_filter = runners.get("observation_filter", "NoFilter")

    if hasattr(config_obj, "env_runners"):
        return config_obj.env_runners(
            num_env_runners=num_runners,
            num_envs_per_env_runner=envs_per_runner,
            rollout_fragment_length=rollout_fragment_length,
            observation_filter=obs_filter,
        )

    return config_obj.rollouts(
        num_rollout_workers=num_runners,
        num_envs_per_worker=envs_per_runner,
        rollout_fragment_length=rollout_fragment_length,
        observation_filter=obs_filter,
    )


def build_rllib_config(experiment_config: dict[str, Any], seed: int | None):
    algo_name = experiment_config["algorithm"].upper()
    config_class = get_algorithm_config_class(algo_name)

    training_config = copy.deepcopy(experiment_config.get("training", {}))
    model_config = copy.deepcopy(experiment_config.get("model", {}))
    evaluation_config = copy.deepcopy(experiment_config.get("evaluation", {}))
    reporting_config = copy.deepcopy(experiment_config.get("reporting", {}))
    exploration_config = copy.deepcopy(experiment_config.get("exploration", {}))
    resources_config = {"num_gpus": 0, "num_gpus_per_worker": 0}
    resources_config.update(experiment_config.get("resources", {}))

    if model_config:
        training_config["model"] = model_config

    config_obj = config_class()

    config_obj = (
        config_obj.environment(env=ENV_NAME, env_config=experiment_config.get("env_config", {}))
        .framework(experiment_config.get("framework", "torch"))
        .training(**training_config)
    )

    if seed is not None:
        config_obj = config_obj.debugging(seed=seed)

    config_obj = configure_env_runners(config_obj, experiment_config)

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
    experiment_name = name or experiment_config.get("name", "drone_rllib_contrib_experiment")
    config_for_build = copy.deepcopy(experiment_config)
    if use_sweep and experiment_config.get("sweep"):
        config_for_build = apply_sweep(config_for_build, experiment_config["sweep"])

    config_obj = build_rllib_config(config_for_build, seed)
    param_space = config_obj.to_dict()

    seeds = experiment_config.get("seeds")
    if seed is None and seeds:
        param_space["seed"] = tune.grid_search(list(seeds))

    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}})
    try:
        tuner = tune.Tuner(
            config_obj.algo_class,
            param_space=param_space,
            tune_config=tune.TuneConfig(num_samples=samples),
            run_config=RunConfig(
                name=experiment_name,
                storage_path=str(results_dir.resolve()),
                stop={STOP_METRIC: stop_steps},
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


def _latest_checkpoint(trial_dir: Path):
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
            try:
                ci = rows[0].split(",").index("training_iteration")
            except ValueError:
                ci = None
            kept = [rows[0]]
            for r in rows[1:]:
                if ci is None:
                    kept.append(r); continue
                try:
                    if int(float(r.split(",")[ci])) <= keep_through_iter:
                        kept.append(r)
                except (ValueError, IndexError):
                    kept.append(r)
            pc.write_text("\n".join(kept) + "\n", encoding="utf-8")


def _resume_one_trial(trial_dir: Path, stop_steps: int, checkpoint_frequency: int) -> None:
    from ray.rllib.algorithms.algorithm import Algorithm

    ckpt = _latest_checkpoint(trial_dir)
    if ckpt is None:
        typer.secho(f"  {trial_dir.name}: no checkpoint_* - skipping", fg=typer.colors.YELLOW)
        return
    algo = Algorithm.from_checkpoint(str(ckpt))
    resume_iter = int(algo.iteration)
    _truncate_logs_to_iter(trial_dir, resume_iter)
    result_file = trial_dir / "result.json"

    def _steps(res):
        return int(res.get(STOP_METRIC) or res.get("timesteps_total") or 0)

    def _save(it):
        target = trial_dir / f"checkpoint_{it:06d}"
        target.mkdir(parents=True, exist_ok=True)
        try:
            algo.save_to_path(str(target))
        except (RuntimeError, AttributeError):
            algo.save(str(target))

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


@app.command()
def resume(
    config: Path = typer.Option(..., "--config", "-c", help="The JSON config the experiment was launched with."),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir", help="Where the experiment dir lives."),
    name: str | None = typer.Option(None, "--name", "-n", help="Experiment name override (defaults to the config's 'name')."),
    steps: int | None = typer.Option(None, "--steps", "-s", help="Target total timesteps (defaults to the config's stop)."),
    errored: bool = typer.Option(False, "--errored", help="Also resume ERRORED trials from their last checkpoint."),
    trial: str | None = typer.Option(None, "--trial", help="Resume ONLY the trial(s) whose folder name contains this text."),
):
    """Continue an interrupted TD3/DDPG run: all unfinished trials, or just one with --trial."""
    import ray
    from ray import tune

    experiment_config = load_experiment_config(config)
    algo_name = experiment_config["algorithm"].upper()
    if algo_name not in SUPPORTED_ALGORITHMS:
        raise typer.BadParameter(f"{config} is {algo_name!r}; this script only handles {sorted(SUPPORTED_ALGORITHMS)}.")

    experiment_name = name or experiment_config.get("name", "drone_rllib_contrib_experiment")
    experiment_dir = (results_dir / experiment_name).resolve()
    if not experiment_dir.exists():
        raise typer.BadParameter(f"No experiment to resume at {experiment_dir}")
    stop_steps = steps or int(experiment_config.get("stop", {}).get("timesteps_total", 100_000))
    checkpoint_frequency = int(experiment_config.get("checkpoint_frequency", 10))

    register_drone_env()
    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}})
    try:
        if trial:
            matches = sorted(d for d in experiment_dir.iterdir() if d.is_dir() and trial in d.name)
            if not matches:
                raise typer.BadParameter(f"No trial dir under {experiment_dir} contains {trial!r}")
            typer.secho(f"Resuming {len(matches)} trial(s) matching {trial!r}:", fg=typer.colors.GREEN)
            for trial_dir in matches:
                _resume_one_trial(trial_dir, stop_steps, checkpoint_frequency)
        else:
            config_obj = get_algorithm_config_class(algo_name)()
            tuner = tune.Tuner.restore(
                str(experiment_dir),
                trainable=config_obj.algo_class,
                resume_unfinished=True,
                resume_errored=errored,
            )
            result_grid = tuner.fit()
            raise_for_failed_trials(result_grid)
    finally:
        ray.shutdown()
    typer.secho(f"{algo_name} resume finished (rllib_contrib).", fg=typer.colors.GREEN)


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", help="Path to a TD3/DDPG RLlib experiment JSON config."),
    steps: int | None = typer.Option(None, "--steps", "-s", help="Override total timesteps."),
    name: str | None = typer.Option(None, "--name", "-n", help="Override experiment name."),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir", help="Where Ray Tune writes results."),
    samples: int = typer.Option(1, "--samples", help="Number of Tune samples for stochastic/randomized sweeps."),
    seed: int | None = typer.Option(None, "--seed", help="Pin ONE seed (disables the config's multi-seed grid)."),
    seeds: str | None = typer.Option(None, "--seeds", help="Comma-separated seed list, overrides the config's 'seeds' (e.g. --seeds 0,1,2)."),
    use_sweep: bool = typer.Option(True, "--sweep/--no-sweep", help="Enable or disable grid-search sweep values from the config."),
):
    """Train one TD3 or DDPG experiment from a JSON config, via rllib_contrib."""
    experiment_config = load_experiment_config(config)
    if seeds is not None:
        experiment_config["seeds"] = [int(s) for s in seeds.split(",") if s.strip() != ""]
    algo = experiment_config["algorithm"].upper()
    if algo not in SUPPORTED_ALGORITHMS:
        raise typer.BadParameter(
            f"{config} declares algorithm={algo!r}. This script only trains "
            f"{sorted(SUPPORTED_ALGORITHMS)}; use ../rllib_train.py for the rest."
        )
    run_tuner(
        experiment_config,
        steps=steps,
        name=name,
        results_dir=results_dir,
        samples=samples,
        seed=seed,
        use_sweep=use_sweep,
    )
    typer.secho(f"{algo} training finished (rllib_contrib).", fg=typer.colors.GREEN)


@app.command()
def batch(
    config_dir: Path = typer.Option(Path("../../AgentnoBaziraniSistemiTD3DDPG/rllib_configs"), "--config-dir", help="Directory with RLlib JSON configs."),
    steps: int | None = typer.Option(None, "--steps", "-s", help="Override total timesteps for every config."),
    seeds: str = typer.Option("0,1,2", "--seeds", help="Comma-separated seeds to run for every config."),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir", help="Where Ray Tune writes results."),
):
    """Run every TD3/DDPG config in a directory across multiple seeds.

    Non-TD3/DDPG configs in the directory (PPO, SAC) are skipped — run those
    through ../rllib_train.py in the main virtualenv instead."""
    config_paths = sorted(config_dir.glob("*.json"))
    if not config_paths:
        raise typer.BadParameter(f"No JSON configs found in {config_dir}")

    parsed_seeds = [int(seed.strip()) for seed in seeds.split(",") if seed.strip()]
    ran_any = False
    for config_path in config_paths:
        base_config = load_experiment_config(config_path)
        algo = base_config.get("algorithm", "").upper()
        if algo not in SUPPORTED_ALGORITHMS:
            typer.echo(f"Skipping {config_path.name} (algorithm={algo!r}, not TD3/DDPG)")
            continue
        ran_any = True
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
    if not ran_any:
        typer.secho(f"No TD3/DDPG configs found in {config_dir}.", fg=typer.colors.YELLOW)
    else:
        typer.secho("rllib_contrib TD3/DDPG comparison finished.", fg=typer.colors.GREEN)


@app.command("show-config")
def show_config(config: Path = typer.Argument(..., help="Path to a TD3/DDPG RLlib experiment JSON config.")):
    """Print a normalized config before training."""
    experiment_config = load_experiment_config(config)
    typer.echo(json.dumps(experiment_config, indent=2))


if __name__ == "__main__":
    app()