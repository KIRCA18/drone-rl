import time
import typer
from enum import Enum

from analyze_rllib_results import app as rllib_results_app
from rllib_train import app as rllib_app

app = typer.Typer(help="Drone RL Comparative Analysis CLI")
app.add_typer(rllib_app, name="rllib", help="Train and compare Ray RLlib experiments.")
app.add_typer(rllib_results_app, name="rllib-results", help="Summarize Ray Tune/RLlib results.")



class Algorithm(str, Enum):
    ppo = "PPO"
    sac = "SAC"
    td3 = "TD3"


def get_algo_class(algo_name: Algorithm):
    """Helper function to map the string input to the actual SB3 class."""
    from stable_baselines3 import PPO, SAC, TD3

    if algo_name == Algorithm.ppo:
        return PPO
    elif algo_name == Algorithm.sac:
        return SAC
    elif algo_name == Algorithm.td3:
        return TD3


@app.command()
def train(
        algo: Algorithm = typer.Option(Algorithm.ppo, "--algo", "-a", help="RL algorithm to use (PPO, SAC, TD3)"),
        steps: int = typer.Option(1_000_000, "--steps", "-s", help="Total timesteps to train"),
        output: str = typer.Option(f"models/{time.time_ns()}", "--output", "-o", help="Name of the saved model file"),
        envs: int = typer.Option(4, "--envs", "-e", help="Number of parallel environments (SAC and TD3 are heavier and prefer single envs)")
):
    """Train a drone policy and save it to disk."""
    from stable_baselines3.common.env_util import make_vec_env
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.enums import ActionType, ObservationType

    if not output.endswith(".zip"):
        output += ".zip"

    typer.echo(f"--- Starting {algo.value} training for {steps} timesteps ---")

    env_kwargs = {"obs": ObservationType('kin'), "act": ActionType('one_d_rpm')}

    envs = envs if algo == Algorithm.ppo else 1

    train_env = make_vec_env(HoverAviary, env_kwargs=env_kwargs, n_envs=envs, seed=0)

    algo_class = get_algo_class(algo)
    model = algo_class("MlpPolicy", train_env, verbose=1)

    model.learn(total_timesteps=steps)
    model.save(output)

    train_env.close()
    typer.secho(f"Training complete! Model saved as {output}", fg=typer.colors.GREEN)


@app.command()
def render(
        algo: Algorithm = typer.Option(..., "--algo", "-a", help="Algorithm used to train the model"),
        model_path: str = typer.Option(..., "--model", "-m", help="Path to the saved model (without .zip)")
):
    """Load a saved model and render it visually in PyBullet."""
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.enums import ActionType, ObservationType

    if not model_path.endswith(".zip"):
        model_path += ".zip"
    typer.echo(f"--- Loading {algo.value} model from {model_path} ---")

    eval_env = HoverAviary(
        gui=True,
        obs=ObservationType('kin'),
        act=ActionType('one_d_rpm')
    )

    algo_class = get_algo_class(algo)
    model = algo_class.load(model_path)

    obs, info = eval_env.reset()
    typer.secho("Launching PyBullet GUI... Press Ctrl+C in terminal to stop.", fg=typer.colors.CYAN)

    try:
        for _ in range(2400):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            time.sleep(1. / 240.)

            if terminated or truncated:
                obs, info = eval_env.reset()
    except KeyboardInterrupt:
        typer.echo("\nEvaluation stopped by user.")
    finally:
        eval_env.close()


@app.command()
def train_and_render(
        algo: Algorithm = typer.Option(Algorithm.ppo, "--algo", "-a", help="RL algorithm to use"),
        steps: int = typer.Option(500_000, "--steps", "-s", help="Total timesteps to train"),
        output: str = typer.Option(f"models/{time.time_ns()}", "--output", "-o", help="Name of the saved model file"),
        envs: int = typer.Option(4, "--envs", "-e", help="Number of parallel environments (SAC and TD3 are heavier and prefer single envs)")
):
    """Train a model and immediately launch the visualizer when finished."""
    train(algo=algo, steps=steps, output=output, envs=envs)

    typer.echo("\nTransitioning to evaluation phase...\n")
    render(algo=algo, model_path=output)


if __name__ == "__main__":
    app()
