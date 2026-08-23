# RLlib Algorithm Comparison

Summary CSV: `experiments\results\summary.csv`
Reward plot: `experiments\plots\reward_comparison.png`

| Rank | Trial | Final reward | Best reward | Timesteps |
| --- | --- | ---: | ---: | ---: |
| 1 | `sac_hover_baseline_baseline_[256, 256]_SAC_DroneHover-v0_dc976_00000_0_2026-08-23_14-33-36` | 468.488 | 471.819 | 100002 |
| 2 | `ppo_hover_baseline_baseline_[256, 256]_PPO_DroneHover-v0_fb46c_00000_0_2026-08-23_13-44-22` | 423.175 | 434.669 | 100000 |
| 3 | `sac_hover_smoke_replay_baseline_[256, 256]_SAC_DroneHover-v0_19739_00000_0_2026-08-23_14-28-09` | 340.109 | 340.109 | 1109 |
| 4 | `ppo_hover_smoke_flat_baseline_[256, 256]_PPO_DroneHover-v0_c283e_00000_0_2026-08-23_13-35-37` | 328.700 | 328.700 | 4000 |
| 5 | `ppo_hover_baseline_baseline_[256, 256]_PPO_DroneHover-v0_64b7f_00000_0_2026-08-23_14-01-38` | 318.890 | 439.108 | 100000 |

## Notes

- Compare final reward together with best reward; a high best reward with a lower final reward can indicate unstable training.
- Re-run the strongest configurations with multiple seeds before treating a result as conclusive.
- Use the noise and offset-start scenarios to check whether a policy generalizes beyond the baseline hover setup.
