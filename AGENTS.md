# Repository Guidelines

## Project Structure & Module Organization

This repository implements AerialVLN, a UAV vision-and-language navigation research codebase. Core training and evaluation entry points live in `src/vlnce_src/`, with shared argument parsing in `src/common/param.py`. Model definitions are in `Model/`, including policies, trainers, encoders, and model utilities. Simulator integration is under `airsim_plugin/`, while shared runtime helpers are in `utils/`. Shell wrappers and dataset download helpers are in `scripts/`. Documentation is in `README.md`, `README-ZH.md`, and `docs/`; static figures are in `files/`. Large runtime assets are expected outside the repo under sibling directories such as `../DATA` and `../ENVs`.

## Build, Test, and Development Commands

Create the expected environment with Python 3.8:

```bash
conda create -n AirVLN python=3.8
conda activate AirVLN
pip install pip==24.0 setuptools==63.2.0
pip install -r requirements.txt
pip install airsim==1.7.0
```

Use `bash scripts/download_dataset_aerialvln.sh` or `bash scripts/download_dataset_aerialvln-s.sh` to fetch dataset files. Run `bash scripts/train.sh` for the default training flow, `bash scripts/eval.sh` for checkpoint evaluation, and `bash scripts/collect.sh` for trajectory collection. These wrappers assume they are launched from the workspace containing `AirVLN`, `DATA`, and `ENVs`.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and keep imports grouped by standard library, third-party packages, then local modules. Follow existing argument names and experiment flags, which currently use mixed project conventions such as `--batchSize`, `--EVAL_DATASET`, and `--trainer_gpu_device`. Prefer descriptive snake_case for new functions, variables, and Python modules. Keep generated files, checkpoints, simulator binaries, datasets, and logs out of the repository.

## Testing Guidelines

No formal unit test suite is currently present. Validate changes with the smallest relevant smoke run: import touched modules, run modified scripts with reduced options such as `--batchSize 1`, and use `--EVAL_NUM` to limit evaluation volume. Changes affecting simulator communication should start `airsim_plugin/AirVLNSimulatorServerTool.py` and verify the client path reaches at least one scene without timeout.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, for example `Fix cloud_eval import path for direct script execution` and `Log eval output directories at startup`. Keep commit subjects concise and action-oriented. Pull requests should describe the affected workflow, list required data/checkpoint assumptions, include the exact commands used for verification, and attach screenshots or log excerpts when changing simulator, evaluation, or cloud-demo behavior.

## Security & Configuration Tips

Do not commit private dataset links, cloud credentials, machine-specific paths, or generated `DATA/output` artifacts. Default simulator ports include `30000` and DDP uses `20000`; document any port changes in the PR when debugging multi-process runs.
