# Cloud LLM Closed-Loop Demo

This demo runs one `AirVLN` episode and replaces local policy inference with cloud LLM inference.

## What it does

1. Connects to simulator and resets one episode (`batch_size=1`).
2. Reads simulator observation (`rgb/depth/pose/progress`).
3. Builds observation summary JSON and sends it to OpenAI-compatible API.
4. Parses cloud action and executes it in simulator.
5. Repeats until done or `--max_steps`.
6. Prints final evaluation metrics.

## Prerequisites

- Start AirVLN simulator server first (example):

```bash
python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0
```

- Install OpenAI SDK:

```bash
pip install openai
```

- Set API key:

```bash
export DASHSCOPE_API_KEY="sk-xxx"
```

## Run demo

```bash
python -u ./scripts/cloud_closed_loop_demo.py \
  --run_type eval \
  --collect_type TF \
  --policy_type seq2seq \
  --name cloud-demo \
  --batchSize 1 \
  --split val_unseen \
  --max_steps 30 \
  --model qwen3.5-flash \
  --enable_thinking
```

> Notes:
>
> - `--run_type/--collect_type/--policy_type` are parsed by existing project args.
> - Demo itself does not load local policy checkpoint.
> - For deterministic output, ask cloud model to return strict JSON like `{"action":"MOVE_FORWARD"}`.

## Git rollback safety

Before experiments, create a branch and commit frequently:

```bash
git checkout -b feat/cloud-closed-loop-demo
git add scripts/cloud_closed_loop_demo.py docs/cloud_closed_loop_demo.md
git commit -m "Add cloud LLM closed-loop demo for AirVLN"
```

Rollback options:

```bash
# Undo working tree changes to last commit
git restore .

# Revert one commit (safe, keeps history)
git revert <commit_hash>

# Hard reset to previous commit (destructive)
git reset --hard HEAD~1
```
