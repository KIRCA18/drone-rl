import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(help="Summarize and plot RLlib/Ray Tune Drone-RL experiment results.")

DEFAULT_RESULTS_DIR = Path("experiments/results")
DEFAULT_REPORT_DIR = Path("reports")
DEFAULT_SUMMARY_DIR = Path("experiments/results")
DEFAULT_PLOT_DIR = Path("experiments/plots")


def parse_number(value: str | None) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_progress(progress_path: Path) -> list[dict[str, Any]]:
    with progress_path.open("r", encoding="utf-8", newline="") as progress_file:
        return list(csv.DictReader(progress_file))


def get_metric(row: dict[str, Any], candidates: list[str]) -> float | None:
    for candidate in candidates:
        value = parse_number(row.get(candidate))
        if value is not None:
            return value
    return None


def trial_name(trial_dir: Path) -> str:
    params_path = trial_dir / "params.json"
    if not params_path.exists():
        return trial_dir.name

    try:
        with params_path.open("r", encoding="utf-8") as params_file:
            params = json.load(params_file)
    except json.JSONDecodeError:
        return trial_dir.name

    algorithm = params.get("algo_class") or params.get("algorithm") or trial_dir.parent.name
    env_config = params.get("env_config", {})
    model = params.get("model", {})
    scenario = env_config.get("scenario", "baseline")
    hidden = model.get("fcnet_hiddens", "model")
    return f"{algorithm}_{scenario}_{hidden}_{trial_dir.name}"


def collect_trials(results_dir: Path) -> list[dict[str, Any]]:
    trials = []
    for progress_path in results_dir.rglob("progress.csv"):
        rows = read_progress(progress_path)
        if not rows:
            continue

        rewards = [
            reward
            for row in rows
            if (reward := get_metric(row, ["episode_reward_mean", "env_runners/episode_return_mean"])) is not None
        ]
        lengths = [
            length
            for row in rows
            if (length := get_metric(row, ["episode_len_mean", "env_runners/episode_len_mean"])) is not None
        ]
        timesteps = [
            step
            for row in rows
            if (step := get_metric(row, ["timesteps_total", "num_env_steps_sampled_lifetime"])) is not None
        ]

        if not rewards:
            continue

        trial_dir = progress_path.parent
        trials.append(
            {
                "trial": trial_name(trial_dir),
                "path": str(trial_dir),
                "iterations": len(rows),
                "timesteps": int(timesteps[-1]) if timesteps else "",
                "best_reward": max(rewards),
                "final_reward": rewards[-1],
                "final_episode_length": lengths[-1] if lengths else "",
                "progress": [(timesteps[index] if index < len(timesteps) else index, reward) for index, reward in enumerate(rewards)],
            }
        )
    return sorted(trials, key=lambda item: item["final_reward"], reverse=True)


def write_summary_csv(trials: list[dict[str, Any]], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=["trial", "iterations", "timesteps", "best_reward", "final_reward", "final_episode_length", "path"],
        )
        writer.writeheader()
        for trial in trials:
            writer.writerow({key: trial[key] for key in writer.fieldnames})


def write_report(trials: list[dict[str, Any]], report_path: Path, summary_path: Path, plot_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RLlib Algorithm Comparison",
        "",
        f"Summary CSV: `{summary_path}`",
        f"Reward plot: `{plot_path}`",
        "",
        "| Rank | Trial | Final reward | Best reward | Timesteps |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for rank, trial in enumerate(trials, start=1):
        lines.append(
            f"| {rank} | `{trial['trial']}` | {trial['final_reward']:.3f} | "
            f"{trial['best_reward']:.3f} | {trial['timesteps']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Compare final reward together with best reward; a high best reward with a lower final reward can indicate unstable training.",
            "- Re-run the strongest configurations with multiple seeds before treating a result as conclusive.",
            "- Use the noise and offset-start scenarios to check whether a policy generalizes beyond the baseline hover setup.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reward_plot(trials: list[dict[str, Any]], plot_path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    for trial in trials[:10]:
        xs = [point[0] for point in trial["progress"]]
        ys = [point[1] for point in trial["progress"]]
        plt.plot(xs, ys, label=trial["trial"][:80])

    plt.xlabel("Timesteps")
    plt.ylabel("Mean episode reward")
    plt.title("RLlib Drone-RL Reward Comparison")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


@app.command()
def summarize(
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir", help="Ray Tune results directory."),
    summary_path: Path | None = typer.Option(None, "--summary", help="Output CSV summary path. Defaults to a timestamped file so old runs aren't overwritten."),
    report_path: Path | None = typer.Option(None, "--report", help="Output Markdown report path. Defaults to a timestamped file so old runs aren't overwritten."),
    plot_path: Path | None = typer.Option(None, "--plot", help="Output reward plot path. Defaults to a timestamped file so old runs aren't overwritten."),
):
    """Create a CSV, Markdown report, and reward curve plot from Ray Tune results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if summary_path is None:
        summary_path = DEFAULT_SUMMARY_DIR / f"summary_{timestamp}.csv"
    if report_path is None:
        report_path = DEFAULT_REPORT_DIR / f"algorithm_comparison_{timestamp}.md"
    if plot_path is None:
        plot_path = DEFAULT_PLOT_DIR / f"reward_comparison_{timestamp}.png"

    trials = collect_trials(results_dir)
    if not trials:
        raise typer.BadParameter(f"No completed Ray Tune progress.csv files with reward metrics found in {results_dir}")

    write_summary_csv(trials, summary_path)
    write_reward_plot(trials, plot_path)
    write_report(trials, report_path, summary_path, plot_path)

    typer.secho(f"Wrote {summary_path}", fg=typer.colors.GREEN)
    typer.secho(f"Wrote {plot_path}", fg=typer.colors.GREEN)
    typer.secho(f"Wrote {report_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()