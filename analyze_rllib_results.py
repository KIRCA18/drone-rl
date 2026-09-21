import csv
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(help="Summarize and plot RLlib/Ray Tune Drone-RL results.")

DEFAULT_RESULTS_DIR = Path("experiments/results")
DEFAULT_REPORT_DIR = Path("reports")
DEFAULT_SUMMARY_DIR = Path("experiments/results")
DEFAULT_PLOT_DIR = Path("experiments/plots")
DEFAULT_ANALYSIS_DIR = Path("experiments/analysis")

# result.json keys, most-preferred first.
EVAL_RETURN_KEYS = [
    ("evaluation", "env_runners", "episode_return_mean"),
    ("evaluation", "env_runners", "episode_reward_mean"),
    ("evaluation", "episode_return_mean"),
    ("evaluation", "episode_reward_mean"),
]
TRAIN_RETURN_KEYS = [
    ("env_runners", "episode_return_mean"),
    ("env_runners", "episode_reward_mean"),
    ("episode_reward_mean",),
    ("sampler_results", "episode_reward_mean"),
]
EVAL_LEN_KEYS = [
    ("evaluation", "env_runners", "episode_len_mean"),
    ("evaluation", "episode_len_mean"),
]
STEP_KEYS = [("timesteps_total",), ("num_env_steps_sampled_lifetime",), ("counters", "num_env_steps_sampled")]


def parse_number(value: str | None) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dig(row: dict, path: tuple):
    cur = row
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, (int, float)) and not (isinstance(cur, float) and math.isnan(cur)):
        return float(cur)
    return None


def _first(row: dict, paths: list[tuple]):
    for p in paths:
        v = _dig(row, p)
        if v is not None:
            return v
    return None


def _load_result_json(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _tokens_from_name(trial_dir_name: str) -> dict[str, str]:
    m = re.search(r"_[0-9a-f]{5}_\d{5}_\d+_([A-Za-z0-9_.,=\[\]\- ]+?)_20\d\d-\d\d-\d\d", trial_dir_name)
    if not m:
        return {}
    out: dict[str, str] = {}
    for tok in m.group(1).split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip().split(".")[-1]] = v.strip()
    return out


def _sweep_param_from_name(trial_dir_name: str) -> tuple[str | None, str | None]:
    """The swept parameter = every name token except `seed`."""
    toks = {k: v for k, v in _tokens_from_name(trial_dir_name).items() if k != "seed"}
    if not toks:
        return None, None
    if len(toks) == 1:
        (k, v), = toks.items()
        return k, v
    return "+".join(sorted(toks)), ",".join(f"{k}={toks[k]}" for k in sorted(toks))


def _sweep_param_from_params(params: dict) -> tuple[str | None, str | None]:
    env_config = params.get("env_config") if isinstance(params.get("env_config"), dict) else {}
    reward_config = env_config.get("reward") if isinstance(env_config.get("reward"), dict) else {}
    reward_mode = reward_config.get("mode")
    if reward_mode and reward_mode != "baseline":
        return "reward.mode", str(reward_mode)

    scenario = env_config.get("scenario")
    if scenario and scenario != "baseline":
        return "scenario", str(scenario)

    noise = env_config.get("observation_noise_std")
    if noise not in (None, 0, 0.0, "0", "0.0"):
        return "observation_noise_std", str(noise)

    return None, None


def _bootstrap_ci(values: list[float], n_boot: int = 5000, alpha: float = 0.05) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0], values[0]) if values else (float("nan"), float("nan"))
    rng = random.Random(0)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int((alpha / 2) * n_boot)], means[int((1 - alpha / 2) * n_boot)]


def _iqm(values: list[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    k = n // 4
    trimmed = s[k : n - k] if n >= 4 else s
    return sum(trimmed) / len(trimmed)


def _mean(v):
    return sum(v) / len(v) if v else float("nan")


def _std(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def _try_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.inf


def collect_trials(results_dir: Path) -> list[dict[str, Any]]:
    trials = []
    for result_path in results_dir.rglob("result.json"):
        trial_dir = result_path.parent
        params_path = trial_dir / "params.json"
        params = {}
        if params_path.exists():
            try:
                params = json.loads(params_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        rows = _load_result_json(result_path)
        if not rows:
            continue

        eval_curve: list[tuple[float, float]] = []
        metric_source = None
        last_eval_len = None
        for r in rows:
            ev = _first(r, EVAL_RETURN_KEYS)
            if ev is None:
                continue
            step = _first(r, STEP_KEYS)
            eval_curve.append((step if step is not None else len(eval_curve), ev))
            metric_source = "evaluation"
            el = _first(r, EVAL_LEN_KEYS)
            if el is not None:
                last_eval_len = el

        if not eval_curve:
            for r in rows:
                tr = _first(r, TRAIN_RETURN_KEYS)
                if tr is None:
                    continue
                step = _first(r, STEP_KEYS)
                eval_curve.append((step if step is not None else len(eval_curve), tr))
            metric_source = "train_return(no_eval_logged)"
        if not eval_curve:
            continue

        eval_curve.sort(key=lambda t: t[0])
        returns = [v for _, v in eval_curve]
        tail = max(1, len(returns) // 5)  # last 20%
        final_perf = _mean(returns[-tail:])
        best_perf = max(returns)

        eval_len = last_eval_len if last_eval_len is not None else _first(rows[-1], EVAL_LEN_KEYS)
        total_steps = _first(rows[-1], STEP_KEYS) or eval_curve[-1][0]

        exp_name = trial_dir.parent.name
        p_name, p_val = _sweep_param_from_name(trial_dir.name)
        if p_name is None:
            p_name, p_val = _sweep_param_from_params(params)
        algo = params.get("algo_class") or params.get("algorithm") or exp_name.split("_")[0].upper()
        seed = params.get("seed")
        if seed is None:
            seed = _tokens_from_name(trial_dir.name).get("seed")

        trials.append(
            {
                "trial": trial_dir.name,
                "experiment": exp_name,
                "trial_dir": str(trial_dir),
                "algo": str(algo).replace("Config", "").upper(),
                "seed": seed,
                "param_name": p_name,
                "param_value": p_val,
                "iterations": len(rows),
                "n_eval_points": len(returns),
                "total_steps": int(total_steps),
                "final_perf": final_perf,
                "best_perf": best_perf,
                "final_eval_len": eval_len,
                "metric_source": metric_source,
                "curve": eval_curve,
            }
        )
    return trials


def aggregate_by_config(trials: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for t in trials:
        groups[(t["experiment"], t["algo"], t["param_name"], t["param_value"])].append(t)

    out = []
    for (exp, algo, pname, pval), ts in sorted(groups.items(), key=lambda kv: str(kv[0])):
        finals = [t["final_perf"] for t in ts]
        lo, hi = _bootstrap_ci(finals)
        lens = [t["final_eval_len"] for t in ts if t["final_eval_len"] is not None]
        out.append(
            {
                "experiment": exp,
                "algo": algo,
                "param_name": pname or "(baseline)",
                "param_value": pval or "-",
                "n_seeds": len(ts),
                "final_mean": _mean(finals),
                "final_std": _std(finals),
                "final_iqm": _iqm(finals),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "final_min": min(finals),
                "final_max": max(finals),
                "eval_len_mean": _mean(lens) if lens else float("nan"),
                "metric_source": ts[0]["metric_source"],
                "seeds": ",".join(str(t["seed"]) for t in ts),
            }
        )
    return out


def _interp_curves(ts: list[dict], n_grid: int = 60):
    xmax = min(max(s for s, _ in t["curve"]) for t in ts)
    xmin = max(min(s for s, _ in t["curve"]) for t in ts)
    if xmax <= xmin:
        xmin, xmax = 0, xmax or 1
    grid = [xmin + (xmax - xmin) * i / (n_grid - 1) for i in range(n_grid)]
    stacked = []
    for t in ts:
        xs = [s for s, _ in t["curve"]]
        ys = [v for _, v in t["curve"]]
        row = []
        for g in grid:
            if g <= xs[0]:
                row.append(ys[0])
            elif g >= xs[-1]:
                row.append(ys[-1])
            else:
                j = next(i for i in range(1, len(xs)) if xs[i] >= g)
                f = (g - xs[j - 1]) / (xs[j] - xs[j - 1]) if xs[j] != xs[j - 1] else 0.0
                row.append(ys[j - 1] + f * (ys[j] - ys[j - 1]))
        stacked.append(row)
    mean = [_mean([r[i] for r in stacked]) for i in range(n_grid)]
    los, his = [], []
    for i in range(n_grid):
        lo, hi = _bootstrap_ci([r[i] for r in stacked], n_boot=1000)
        los.append(lo)
        his.append(hi)
    return grid, mean, los, his


def _plot_experiment(exp: str, trials: list[dict], plot_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    by_val: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        by_val[t["param_value"] or "-"].append(t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    labels, means, errs = [], [], []
    for val, ts in sorted(by_val.items(), key=lambda kv: _try_float(kv[0])):
        grid, mean, lo, hi = _interp_curves(ts)
        ax1.plot(grid, mean, label=f"{ts[0]['param_name'] or 'cfg'}={val} (n={len(ts)})")
        ax1.fill_between(grid, lo, hi, alpha=0.2)
        finals = [t["final_perf"] for t in ts]
        clo, chi = _bootstrap_ci(finals)
        labels.append(str(val))
        means.append(_mean(finals))
        errs.append([_mean(finals) - clo, chi - _mean(finals)])

    ax1.set_xlabel("environment steps")
    ax1.set_ylabel("evaluation return (mean +/- 95% CI)")
    ax1.set_title(f"{exp}: learning curves")
    ax1.legend(fontsize="small")
    ax1.grid(alpha=0.3)

    xs = list(range(len(labels)))
    ax2.bar(xs, means, yerr=list(zip(*errs)) if errs else None, capsize=4)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("final eval return (last 20%, 95% CI)")
    ax2.set_title(f"{exp}: final performance")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"{exp}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# summarize  --  ORIGINAL interface (--results-dir / --summary / --report / --plot)
# --------------------------------------------------------------------------- #
def write_summary_csv(trials: list[dict], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["trial", "iterations", "timesteps", "best_reward", "final_reward",
              "final_episode_length", "algo", "seed", "param", "metric_source", "path"]
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for t in trials:
            w.writerow({
                "trial": t["trial"],
                "iterations": t["iterations"],
                "timesteps": t["total_steps"],
                "best_reward": round(t["best_perf"], 3),
                "final_reward": round(t["final_perf"], 3),
                "final_episode_length": "" if t["final_eval_len"] is None else round(t["final_eval_len"], 1),
                "algo": t["algo"],
                "seed": t["seed"],
                "param": f"{t['param_name']}={t['param_value']}" if t["param_name"] else "",
                "metric_source": t["metric_source"],
                "path": t["trial_dir"],
            })


def write_report(trials: list[dict], report_path: Path, summary_path: Path, plot_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RLlib Algorithm Comparison",
        "",
        f"Summary CSV: `{summary_path}`",
        f"Reward plot: `{plot_path}`",
        "",
        "_Reward = deterministic evaluation return (`explore=False`). "
        "Final = mean over the last 20% of the run._",
        "",
        "| Rank | Trial | Final reward | Best reward | Timesteps | Metric |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for rank, t in enumerate(trials, start=1):
        lines.append(
            f"| {rank} | `{t['trial']}` | {t['final_perf']:.3f} | "
            f"{t['best_perf']:.3f} | {t['total_steps']} | {t['metric_source']} |"
        )

    agg = aggregate_by_config(trials)
    by_exp: dict[str, list[dict]] = defaultdict(list)
    for r in agg:
        by_exp[r["experiment"]].append(r)

    lines += [
        "",
        "## Per-configuration performance (aggregated across seeds)",
        "",
        "_CI = 95% bootstrap over seeds. IQM = interquartile mean (drops the "
        "best and worst seed)._",
    ]
    for exp in sorted(by_exp):
        lines += [
            "",
            f"### {exp}",
            "",
            "| Param | Value | Seeds | Mean | Std | IQM | 95% CI | Eval len |",
            "| --- | --- | ---: | ---: | ---: | ---: | :--- | ---: |",
        ]
        for r in sorted(by_exp[exp], key=lambda r: _try_float(r["param_value"])):
            lines.append(
                f"| {r['param_name']} | {r['param_value']} | {r['n_seeds']} | "
                f"{r['final_mean']:.1f} | {r['final_std']:.1f} | {r['final_iqm']:.1f} | "
                f"[{r['ci95_lo']:.1f}, {r['ci95_hi']:.1f}] | {r['eval_len_mean']:.0f} |"
            )

    lines += [
        "",
        "## Notes",
        "",
        "- Compare final reward with best reward: best ≫ final means the run peaked then collapsed.",
        "- Episode length near the truncation limit (~242) = a stable hover; ~60 = the drone crashes early.",
        "- Use the noise / offset-start scenarios to check whether a policy generalizes beyond baseline hover.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reward_plot(trials: list[dict], plot_path: Path, top_n: int = 10) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    for t in trials[:top_n]:
        xs = [s for s, _ in t["curve"]]
        ys = [v for _, v in t["curve"]]
        plt.plot(xs, ys, label=t["trial"][:80])
    plt.xlabel("environment steps")
    plt.ylabel("evaluation return")
    plt.title(f"RLlib Drone-RL — top {top_n} trials by final eval return")
    plt.legend(fontsize="small")
    plt.grid(alpha=0.3)
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
        raise typer.BadParameter(f"No completed result.json files with reward metrics found in {results_dir}")
    trials.sort(key=lambda t: t["final_perf"], reverse=True)

    write_summary_csv(trials, summary_path)
    write_reward_plot(trials, plot_path)
    write_report(trials, report_path, summary_path, plot_path)

    typer.secho(f"Wrote {summary_path}", fg=typer.colors.GREEN)
    typer.secho(f"Wrote {plot_path}", fg=typer.colors.GREEN)
    typer.secho(f"Wrote {report_path}", fg=typer.colors.GREEN)


@app.command()
def aggregate(
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir", help="Ray Tune results directory."),
    analysis_dir: Path = typer.Option(DEFAULT_ANALYSIS_DIR, "--analysis-dir", help="Where the per-trial / per-group CSVs go."),
    plot_dir: Path = typer.Option(DEFAULT_PLOT_DIR, "--plot-dir", help="Where per-experiment figures go."),
    report_dir: Path = typer.Option(DEFAULT_REPORT_DIR, "--report-dir", help="Where the Markdown roll-up goes."),
):
    """Per-trial + per-(experiment, swept-value) CSVs, one figure per experiment, Markdown roll-up."""
    trials = collect_trials(results_dir)
    if not trials:
        raise typer.BadParameter(f"No result.json trials with returns found under {results_dir}")

    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    trial_csv = analysis_dir / "trials.csv"
    with trial_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "algo", "seed", "param_name", "param_value",
                    "iterations", "total_steps", "final_perf", "best_perf",
                    "final_eval_len", "metric_source", "trial_dir"])
        for t in sorted(trials, key=lambda t: (t["experiment"], str(t["param_value"]), str(t["seed"]))):
            w.writerow([t["experiment"], t["algo"], t["seed"], t["param_name"], t["param_value"],
                        t["iterations"], t["total_steps"], f"{t['final_perf']:.3f}", f"{t['best_perf']:.3f}",
                        "" if t["final_eval_len"] is None else f"{t['final_eval_len']:.1f}",
                        t["metric_source"], t["trial_dir"]])

    agg = aggregate_by_config(trials)
    agg_csv = analysis_dir / "aggregate.csv"
    with agg_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
        w.writeheader()
        for row in agg:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in row.items()})

    by_exp: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        by_exp[t["experiment"]].append(t)

    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = report_dir / f"aggregate_{ts_now}.md"
    lines = ["# Multi-seed RLlib aggregation", "",
             f"- results: `{results_dir}`  |  trials: {len(trials)}  |  experiments: {len(by_exp)}",
             f"- per-trial: `{trial_csv}`  |  per-group: `{agg_csv}`",
             "- **final performance** = mean deterministic-eval return over the last 20% of each run.",
             "- CI = 95% bootstrap over seeds. IQM = interquartile mean.", ""]
    plots = []
    for exp in sorted(by_exp):
        p = _plot_experiment(exp, by_exp[exp], plot_dir)
        if p:
            plots.append(p)
        rows = [r for r in agg if r["experiment"] == exp]
        src = rows[0]["metric_source"] if rows else "?"
        lines += [f"## {exp}", "", f"_metric: {src}_", "",
                  "| param | value | seeds | final mean | std | IQM | 95% CI | eval len |",
                  "| --- | --- | ---: | ---: | ---: | ---: | :--- | ---: |"]
        for r in sorted(rows, key=lambda r: _try_float(r["param_value"])):
            lines.append(
                f"| {r['param_name']} | {r['param_value']} | {r['n_seeds']} | "
                f"{r['final_mean']:.1f} | {r['final_std']:.1f} | {r['final_iqm']:.1f} | "
                f"[{r['ci95_lo']:.1f}, {r['ci95_hi']:.1f}] | {r['eval_len_mean']:.0f} |"
            )
        if p:
            lines += ["", f"![{exp}](../{p.as_posix()})"]
        lines.append("")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    typer.secho(f"wrote {trial_csv}", fg=typer.colors.GREEN)
    typer.secho(f"wrote {agg_csv}", fg=typer.colors.GREEN)
    typer.secho(f"wrote {report}", fg=typer.colors.GREEN)
    for p in plots:
        typer.secho(f"wrote {p}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
