# Cloud LLM Closed-Loop / Eval

This repository now has two cloud-inference entrypoints:

1. `scripts/cloud_closed_loop_demo.py`: single-episode debug demo.
2. `src/vlnce_src/cloud_eval.py`: eval-style runner (1:1 structure with `train.py --run_type eval`, but action selection is replaced by cloud LLM).

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

## 1) Single-episode debug demo

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

## 2) Eval-style cloud runner (recommended)

This keeps eval directory layout and metrics aggregation behavior aligned with `train.py` eval.

```bash
python -u ./src/vlnce_src/cloud_eval.py \
  --run_type eval \
  --policy_type seq2seq \
  --collect_type TF \
  --name cloud-eval \
  --batchSize 8 \
  --EVAL_CKPT_PATH_DIR ../DATA/output/AirVLN-seq2seq/train/checkpoint \
  --EVAL_DATASET val_unseen \
  --EVAL_NUM -1 \
  --cloud_model qwen3.5-flash \
  --enable_thinking
```

`--EVAL_CKPT_PATH_DIR` is optional for cloud-only eval logic. If omitted (or path
does not exist), runner executes one eval pass with `checkpoint_index=0`.
When eval starts, it prints all output directories (TensorBoard, per-episode,
intermediate, aggregated JSON) to logs.

Output files are written under:

- `DATA/output/{name}/eval/intermediate_results_every/{make_dir_time}/{checkpoint_index}/{episode_id}.json`
- `DATA/output/{name}/eval/intermediate_results/{make_dir_time}/stats_ckpt_{checkpoint}_{split}.json`
- `DATA/output/{name}/eval/results/{make_dir_time}/stats_ckpt_{checkpoint}_{split}.json`

## Notes

- Cloud runner uses retries (`--cloud_max_retries`, default 3) and falls back to `STOP` when cloud inference repeatedly fails.
- For deterministic behavior, force cloud response to strict JSON action format: `{"action":"MOVE_FORWARD"}`.

## Git rollback safety

Before experiments, create a branch and commit frequently:

```bash
git checkout -b feat/cloud-eval
git add src/vlnce_src/cloud_eval.py scripts/cloud_closed_loop_demo.py docs/cloud_closed_loop_demo.md
git commit -m "Add eval-style cloud LLM runner"
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
