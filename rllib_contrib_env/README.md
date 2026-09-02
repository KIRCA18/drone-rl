# `vendor/` — third-party code attribution

The `td3/` and `ddpg/` subdirectories are the **TD3 and DDPG algorithm
implementations, copied verbatim from a third party. This is not our code.**

| | |
|---|---|
| **Source project** | [`ray-project/ray`](https://github.com/ray-project/ray) |
| **Paths taken** | `rllib_contrib/td3/` → `vendor/td3/`  ·  `rllib_contrib/ddpg/` → `vendor/ddpg/` |
| **Version** | git tag **`ray-2.9.0`** — the last release in which `rllib_contrib/td3` and `rllib_contrib/ddpg` exist. Ray deleted the `rllib_contrib/` tree in [ray-project/ray#48565](https://github.com/ray-project/ray/pull/48565) (November 2024). |
| **Authors** | The Ray team and `rllib_contrib` community contributors |
| **License** | **Apache License 2.0** (the license of the `ray-project/ray` repository) |
| **Local modifications** | **None.** The files are unchanged. The only difference from upstream is that `vendor/td3/src` and `vendor/ddpg/src` are added to `sys.path` at runtime by `../rllib_train_contrib.py` (nothing is written to disk). |

Directory layout is kept exactly as upstream so the package imports resolve:

```
vendor/
  td3/src/rllib_td3/td3/      td3.py, __init__.py
  ddpg/src/rllib_ddpg/ddpg/   ddpg.py, ddpg_torch_model.py, ddpg_torch_policy.py,
                              ddpg_tf_model.py, ddpg_tf_policy.py, noop_model.py,
                              utils.py, __init__.py
```

To regenerate this folder from upstream:

```bash
git clone --filter=blob:none --sparse https://github.com/ray-project/ray.git _ray
cd _ray && git sparse-checkout set rllib_contrib/ddpg rllib_contrib/td3 && git checkout ray-2.9.0 && cd ..
rm -rf vendor && mkdir vendor
cp -r _ray/rllib_contrib/ddpg vendor/ddpg
cp -r _ray/rllib_contrib/td3  vendor/td3
rm -rf _ray
```
