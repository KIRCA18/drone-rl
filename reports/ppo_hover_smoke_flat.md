# RLlib Algorithm Comparison

Summary CSV: `experiments\results\ppo_hover_smoke_flat_summary.csv`
Reward plot: `experiments\plots\ppo_hover_smoke_flat.png`

| Rank | Trial | Final reward | Best reward | Timesteps |
| --- | --- | ---: | ---: | ---: |
| 1 | `ppo_hover_smoke_flat_baseline_[256, 256]_PPO_DroneHover-v0_c283e_00000_0_2026-08-23_13-35-37` | 328.700 | 328.700 | 4000 |

## Notes

- Compare final reward together with best reward; a high best reward with a lower final reward can indicate unstable training.
- Re-run the strongest configurations with multiple seeds before treating a result as conclusive.
- Use the noise and offset-start scenarios to check whether a policy generalizes beyond the baseline hover setup.
