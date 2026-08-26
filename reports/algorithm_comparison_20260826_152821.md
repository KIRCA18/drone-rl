# RLlib Algorithm Comparison

Summary CSV: `experiments\results\summary_20260826_152821.csv`
Reward plot: `experiments\plots\reward_comparison_20260826_152821.png`

| Rank | Trial | Final reward | Best reward | Timesteps |
| --- | --- | ---: | ---: | ---: |
| 1 | `ppo_hover_baseline_baseline_[64, 64]_PPO_DroneHover-v0_9eaf0_00000_0_lr=0.0003,fcnet_hiddens=64_64_2026-08-24_14-45-00` | 469.999 | 472.321 | 1000000 |
| 2 | `ppo_hover_fcnet_activation_sweep_baseline_[256, 256]_PPO_DroneHover-v0_30e68_00001_1_fcnet_activation=relu_2026-08-25_21-14-27` | 461.000 | 466.062 | 1000000 |
| 3 | `ppo_hover_clip_param_sweep_baseline_[256, 256]_PPO_DroneHover-v0_5f712_00002_2_clip_param=0.3000_2026-08-25_14-13-26` | 460.757 | 471.490 | 1000000 |
| 4 | `ppo_hover_clip_param_sweep_baseline_[256, 256]_PPO_DroneHover-v0_5f712_00001_1_clip_param=0.2000_2026-08-25_14-13-26` | 459.464 | 467.776 | 1000000 |
| 5 | `ppo_hover_num_epochs_sweep_baseline_[256, 256]_PPO_DroneHover-v0_fdd33_00001_1_num_epochs=10_2026-08-26_12-57-56` | 458.543 | 468.675 | 1000000 |
| 6 | `ppo_hover_baseline_baseline_[64, 64]_PPO_DroneHover-v0_9eaf0_00001_1_lr=0.0001,fcnet_hiddens=64_64_2026-08-24_14-45-00` | 452.716 | 465.845 | 1000000 |
| 7 | `ppo_hover_baseline_baseline_[256, 256]_PPO_DroneHover-v0_9eaf0_00002_2_lr=0.0003,fcnet_hiddens=256_256_2026-08-24_14-45-00` | 449.950 | 468.039 | 1000000 |
| 8 | `ppo_hover_entropy_coeff_sweep_baseline_[256, 256]_PPO_DroneHover-v0_43b5c_00001_1_entropy_coeff=0.0100_2026-08-25_16-07-11` | 448.068 | 455.020 | 1000000 |
| 9 | `ppo_hover_num_epochs_sweep_baseline_[256, 256]_PPO_DroneHover-v0_fdd33_00002_2_num_epochs=20_2026-08-26_12-57-56` | 447.143 | 469.119 | 1000000 |
| 10 | `ppo_hover_clip_param_sweep_baseline_[256, 256]_PPO_DroneHover-v0_5f712_00000_0_clip_param=0.1000_2026-08-25_14-13-25` | 443.968 | 465.892 | 1000000 |
| 11 | `ppo_hover_lambda__sweep_baseline_[256, 256]_PPO_DroneHover-v0_ceefd_00002_2_lambda=0.9900_2026-08-26_11-23-33` | 438.258 | 453.210 | 1000000 |
| 12 | `ppo_hover_gamma_sweep_baseline_[256, 256]_PPO_DroneHover-v0_36445_00002_2_gamma=0.9950_2026-08-25_23-09-08` | 429.697 | 441.424 | 1000000 |
| 13 | `ppo_hover_num_epochs_sweep_baseline_[256, 256]_PPO_DroneHover-v0_fdd33_00000_0_num_epochs=5_2026-08-26_12-57-55` | 421.122 | 423.027 | 1000000 |
| 14 | `ppo_hover_baseline_baseline_[256, 256]_PPO_DroneHover-v0_f7e48_00002_2_lr=0.0003,fcnet_hiddens=256_256_2026-08-24_14-40-20` | 351.236 | 351.236 | 24000 |
| 15 | `ppo_hover_baseline_baseline_[64, 64]_PPO_DroneHover-v0_2bf1d_00000_0_lr=0.0003,fcnet_hiddens=64_64_2026-08-24_14-34-38` | 342.764 | 342.764 | 68000 |
| 16 | `ppo_hover_baseline_baseline_[256, 256]_PPO_DroneHover-v0_ed18d_00003_3_lr=0.0001,fcnet_hiddens=256_256_2026-08-24_14-04-15` | 342.540 | 342.540 | 12000 |
| 17 | `ppo_hover_baseline_baseline_[256, 256]_PPO_DroneHover-v0_f7e48_00003_3_lr=0.0001,fcnet_hiddens=256_256_2026-08-24_14-40-20` | 340.144 | 340.144 | 24000 |
| 18 | `ppo_hover_baseline_baseline_[256, 256]_PPO_DroneHover-v0_ed18d_00002_2_lr=0.0003,fcnet_hiddens=256_256_2026-08-24_14-04-15` | 336.043 | 336.043 | 12000 |
| 19 | `ppo_hover_entropy_coeff_sweep_baseline_[256, 256]_PPO_DroneHover-v0_43b5c_00000_0_entropy_coeff=0.0000_2026-08-25_16-07-10` | 332.834 | 438.673 | 1000000 |
| 20 | `ppo_hover_baseline_baseline_[64, 64]_PPO_DroneHover-v0_f7e48_00001_1_lr=0.0001,fcnet_hiddens=64_64_2026-08-24_14-40-20` | 331.557 | 331.557 | 24000 |
| 21 | `ppo_hover_baseline_baseline_[256, 256]_PPO_DroneHover-v0_b4206_00000_0_2026-08-24_13-48-20` | 328.755 | 328.755 | 4000 |
| 22 | `ppo_hover_baseline_baseline_[64, 64]_PPO_DroneHover-v0_f7e48_00000_0_lr=0.0003,fcnet_hiddens=64_64_2026-08-24_14-40-20` | 326.468 | 326.468 | 24000 |
| 23 | `ppo_hover_baseline_baseline_[64, 64]_PPO_DroneHover-v0_ed18d_00000_0_lr=0.0003,fcnet_hiddens=64_64_2026-08-24_14-04-15` | 306.764 | 314.838 | 12000 |
| 24 | `ppo_hover_baseline_baseline_[256, 256]_PPO_DroneHover-v0_9eaf0_00003_3_lr=0.0001,fcnet_hiddens=256_256_2026-08-24_14-45-00` | 305.875 | 463.663 | 1000000 |
| 25 | `ppo_hover_baseline_baseline_[64, 64]_PPO_DroneHover-v0_ed18d_00001_1_lr=0.0001,fcnet_hiddens=64_64_2026-08-24_14-04-15` | 301.846 | 301.846 | 12000 |
| 26 | `ppo_hover_fcnet_activation_sweep_baseline_[256, 256]_PPO_DroneHover-v0_30e68_00000_0_fcnet_activation=tanh_2026-08-25_21-14-27` | 294.006 | 388.294 | 1000000 |
| 27 | `ppo_hover_gamma_sweep_baseline_[256, 256]_PPO_DroneHover-v0_36445_00001_1_gamma=0.9900_2026-08-25_23-09-08` | 292.775 | 443.597 | 1000000 |
| 28 | `ppo_hover_lambda__sweep_baseline_[256, 256]_PPO_DroneHover-v0_ceefd_00001_1_lambda=0.9500_2026-08-26_11-23-33` | 265.630 | 467.176 | 1000000 |
| 29 | `ppo_hover_lambda__sweep_baseline_[256, 256]_PPO_DroneHover-v0_ceefd_00000_0_lambda=0.9000_2026-08-26_11-23-33` | 220.205 | 472.905 | 1000000 |
| 30 | `ppo_hover_entropy_coeff_sweep_baseline_[256, 256]_PPO_DroneHover-v0_43b5c_00002_2_entropy_coeff=0.0500_2026-08-25_16-07-11` | 213.197 | 391.253 | 1000000 |
| 31 | `ppo_hover_gamma_sweep_baseline_[256, 256]_PPO_DroneHover-v0_36445_00000_0_gamma=0.9500_2026-08-25_23-09-08` | 210.229 | 430.785 | 1000000 |

## Notes

- Compare final reward together with best reward; a high best reward with a lower final reward can indicate unstable training.
- Re-run the strongest configurations with multiple seeds before treating a result as conclusive.
- Use the noise and offset-start scenarios to check whether a policy generalizes beyond the baseline hover setup.
