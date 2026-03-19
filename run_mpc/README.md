# `run_mpc/main.py`

Run batched MPC rollouts from a JSON config and save successful trajectories to an HDF5 dataset.

## Prerequisite

This script must be run in the `mjwarp` pixi environment:

```bash
pixi shell -e mjwarp
```

Run commands below from the repository root (`judo/`).

## Basic usage

```bash
python run_mpc/main.py --config-path run_mpc/configs/spot_tire_upright.json
```

By default, this writes output to:

- `run_mpc/configs/trajectories.h5`

## Common examples

Run 50 trajectories with 8 in parallel:

```bash
python run_mpc/main.py \
  --config-path run_mpc/configs/spot_tire_upright.json \
  --num-trajectories 50 \
  --num-parallel 8
```

Choose a custom output file:

```bash
python run_mpc/main.py \
  --config-path run_mpc/configs/spot_tire_upright.json \
  --dataset-output-path outputs/mpc/spot_tire_upright.h5
```

Enable visualization (no parallel, single env only):

```bash
python run_mpc/main.py \
  --config-path run_mpc/configs/spot_tire_upright.json \
  --num-parallel 1 \
  --visualize
```

## Skip success filtering

By default, only trajectories where `task.success()` returns `True` are saved.
For tasks without a meaningful success criterion, use `--no-require-success` to keep all trajectories:

```bash
python run_mpc/main.py \
  --config-path run_mpc/configs/cylinder_push.json \
  --no-require-success
```

> **TODO(dta-bdai):** Not all tasks (especially non-spot tasks) have a `success()` implementation.
> Tasks missing an override inherit `return False` from the base class, which causes 0% success rate.
> Add `success()` to remaining tasks: `cartpole`, `fr3_pick`, `leap_cube`, `leap_cube_down`, `caltech_leap_cube`.

## Visualize the dataset

After generating trajectories, visualize them with:

```bash
python run_mpc/visualize_trajectories.py --dataset-path <dataset-output-path>
```

Replace `<dataset-output-path>` with the path used during generation (default: `run_mpc/configs/trajectories.h5`).

## One-liner without entering shell

```bash
pixi run -e mjwarp python run_mpc/main.py --config-path run_mpc/configs/spot_tire_upright.json
```
