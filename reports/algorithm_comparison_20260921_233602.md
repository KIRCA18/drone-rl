# RLlib Algorithm Comparison

Summary CSV: `experiments\results\summary_20260921_233602.csv`
Reward plot: `experiments\plots\reward_comparison_20260921_233602.png`

_Reward = deterministic evaluation return (`explore=False`). Final = mean over the last 20% of the run._

| Rank | Trial | Final reward | Best reward | Timesteps | Metric |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `SAC_DroneHover-v0_dc976_00000_0_2026-08-23_14-33-36` | 465.269 | 473.951 | 100002 | evaluation |
| 2 | `ARS_DroneHover-v0_8f977_00000_0_seed=0_2026-09-16_23-57-07` | 455.958 | 480.900 | 500186 | train_return(no_eval_logged) |
| 3 | `PPO_DroneHover-v0_fb46c_00000_0_2026-08-23_13-44-22` | 437.789 | 470.401 | 100000 | evaluation |
| 4 | `SAC_2e2e4_00014` | 423.443 | 442.229 | 102400 | evaluation |
| 5 | `SAC_2e2e4_00018` | 421.407 | 440.342 | 102400 | evaluation |
| 6 | `SAC_2e2e4_00017` | 421.344 | 421.344 | 102400 | evaluation |
| 7 | `SAC_2e2e4_00009` | 420.747 | 456.632 | 102400 | evaluation |
| 8 | `SAC_2e2e4_00007` | 418.121 | 418.121 | 102400 | evaluation |
| 9 | `SAC_2e2e4_00006` | 407.482 | 407.482 | 102400 | evaluation |
| 10 | `SAC_2e2e4_00002` | 397.418 | 409.341 | 102400 | evaluation |
| 11 | `SAC_2e2e4_00010` | 390.704 | 431.976 | 102400 | evaluation |
| 12 | `SAC_2e2e4_00001` | 377.776 | 416.592 | 102400 | evaluation |
| 13 | `SAC_2e2e4_00011` | 355.683 | 430.635 | 102400 | evaluation |
| 14 | `SAC_2e2e4_00000` | 353.599 | 353.599 | 102400 | evaluation |
| 15 | `SAC_2e2e4_00003` | 345.970 | 345.970 | 102400 | evaluation |
| 16 | `ARS_DroneHover-v0_2d4dd_00000_0_seed=0_2026-09-16_23-54-23` | 345.667 | 345.667 | 5526 | train_return(no_eval_logged) |
| 17 | `SAC_DroneHover-v0_19739_00000_0_2026-08-23_14-28-09` | 333.855 | 333.855 | 1109 | evaluation |
| 18 | `PPO_DroneHover-v0_c283e_00000_0_2026-08-23_13-35-37` | 328.700 | 328.700 | 4000 | train_return(no_eval_logged) |
| 19 | `PPO_DroneHover-v0_64b7f_00000_0_2026-08-23_14-01-38` | 315.865 | 463.932 | 100000 | evaluation |
| 20 | `PPO_8e0f0_00000` | 296.086 | 296.086 | 4096 | train_return(no_eval_logged) |
| 21 | `PPO_DroneHover-v0_0329f_00000_0_2026-08-23_19-35-20` | 276.607 | 276.607 | 4000 | train_return(no_eval_logged) |
| 22 | `SAC_2e2e4_00013` | 241.975 | 354.684 | 102400 | evaluation |
| 23 | `SAC_2e2e4_00004` | 237.846 | 297.593 | 102400 | evaluation |
| 24 | `SAC_2e2e4_00005` | 215.586 | 394.213 | 102400 | evaluation |
| 25 | `SAC_2e2e4_00015` | 182.908 | 419.461 | 102400 | evaluation |
| 26 | `SAC_2e2e4_00016` | 133.552 | 213.371 | 102400 | evaluation |
| 27 | `SAC_2e2e4_00008` | 101.426 | 265.527 | 102400 | evaluation |
| 28 | `SAC_2e2e4_00019` | 96.828 | 361.060 | 102400 | evaluation |
| 29 | `SAC_2e2e4_00012` | 73.987 | 273.262 | 102400 | evaluation |

## Per-configuration performance (aggregated across seeds)

_CI = 95% bootstrap over seeds. IQM = interquartile mean (drops the best and worst seed)._

### ars_hover_baseline

| Param | Value | Seeds | Mean | Std | IQM | 95% CI | Eval len |
| --- | --- | ---: | ---: | ---: | ---: | :--- | ---: |
| (baseline) | - | 1 | 456.0 | 0.0 | 456.0 | [456.0, 456.0] | nan |

### ars_hover_bounds_smoke

| Param | Value | Seeds | Mean | Std | IQM | 95% CI | Eval len |
| --- | --- | ---: | ---: | ---: | ---: | :--- | ---: |
| (baseline) | - | 1 | 345.7 | 0.0 | 345.7 | [345.7, 345.7] | nan |

### codex_path_smoke

| Param | Value | Seeds | Mean | Std | IQM | 95% CI | Eval len |
| --- | --- | ---: | ---: | ---: | ---: | :--- | ---: |
| reward.mode | stable_hover | 1 | 296.1 | 0.0 | 296.1 | [296.1, 296.1] | nan |

### ppo_hover_baseline

| Param | Value | Seeds | Mean | Std | IQM | 95% CI | Eval len |
| --- | --- | ---: | ---: | ---: | ---: | :--- | ---: |
| (baseline) | - | 2 | 376.8 | 86.2 | 376.8 | [315.9, 437.8] | 212 |

### ppo_hover_noise_smoke_clip

| Param | Value | Seeds | Mean | Std | IQM | 95% CI | Eval len |
| --- | --- | ---: | ---: | ---: | ---: | :--- | ---: |
| observation_noise_std | 0.01 | 1 | 276.6 | 0.0 | 276.6 | [276.6, 276.6] | nan |

### ppo_hover_smoke_flat

| Param | Value | Seeds | Mean | Std | IQM | 95% CI | Eval len |
| --- | --- | ---: | ---: | ---: | ---: | :--- | ---: |
| (baseline) | - | 1 | 328.7 | 0.0 | 328.7 | [328.7, 328.7] | nan |

### sac_hover_baseline

| Param | Value | Seeds | Mean | Std | IQM | 95% CI | Eval len |
| --- | --- | ---: | ---: | ---: | ---: | :--- | ---: |
| (baseline) | - | 1 | 465.3 | 0.0 | 465.3 | [465.3, 465.3] | 242 |

### sac_hover_reward_modes

| Param | Value | Seeds | Mean | Std | IQM | 95% CI | Eval len |
| --- | --- | ---: | ---: | ---: | ---: | :--- | ---: |
| reward.mode | energy_aware | 5 | 408.1 | 14.4 | 408.8 | [396.7, 419.4] | 242 |
| reward.mode | position | 5 | 180.1 | 115.2 | 157.6 | [96.9, 280.0] | 135 |
| reward.mode | smooth_control | 5 | 279.9 | 134.3 | 294.9 | [165.8, 378.7] | 179 |
| reward.mode | stable_hover | 5 | 335.5 | 99.4 | 346.8 | [258.6, 412.4] | 196 |

### sac_hover_smoke_replay

| Param | Value | Seeds | Mean | Std | IQM | 95% CI | Eval len |
| --- | --- | ---: | ---: | ---: | ---: | :--- | ---: |
| (baseline) | - | 1 | 333.9 | 0.0 | 333.9 | [333.9, 333.9] | 242 |

## Notes

- Compare final reward with best reward: best ≫ final means the run peaked then collapsed.
- Episode length near the truncation limit (~242) = a stable hover; ~60 = the drone crashes early.
- Use the noise / offset-start scenarios to check whether a policy generalizes beyond baseline hover.
