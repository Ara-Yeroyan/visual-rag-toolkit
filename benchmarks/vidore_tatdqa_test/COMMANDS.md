# ViDoRe TAT-DQA (Qdrant) — commands

## Environment

Either export:

```bash
export QDRANT_URL="..."
export QDRANT_API_KEY="..."  # optional
```

Or create a `.env` file in `visual-rag-toolkit/` with the same variables.

## Index + evaluate (single run)

This is the “all-in-one” script (indexes, then evaluates once):

```bash
python -m benchmarks.vidore_tatdqa_test.run_qdrant \
  --dataset vidore/tatdqa_test \
  --collection vidore_tatdqa_test \
  --recreate --index \
  --indexing-threshold 0 \
  --batch-size 6 \
  --upload-batch-size 12 \
  --upload-workers 0 \
  --loader-workers 0 \
  --prefer-grpc \
  --torch-dtype float16 \
  --no-upsert-wait \
  --qdrant-vector-dtype float16
```

## Evaluate only (no re-index) — baseline + sweeps

These commands assume the Qdrant collection already exists and is populated.

### Baseline: single-stage full MaxSim

```bash
python -m benchmarks.vidore_tatdqa_test.sweep_eval \
  --dataset vidore/tatdqa_test \
  --collection vidore_tatdqa_test \
  --prefer-grpc \
  --mode single_full \
  --torch-dtype auto \
  --query-batch-size 32 \
  --top-k 10 \
  --out-dir results/sweeps
```

### Two-stage sweep (preferred): stage-1 tokens vs tiles, stage-2 full rerank

```bash
python -m benchmarks.vidore_tatdqa_test.sweep_eval \
  --dataset vidore/tatdqa_test \
  --collection vidore_tatdqa_test \
  --prefer-grpc \
  --mode two_stage \
  --stage1-mode tokens_vs_tiles \
  --prefetch-ks 20,50,100,200,400 \
  --torch-dtype auto \
  --query-batch-size 32 \
  --top-k 10 \
  --out-dir results/sweeps
```

### Smoke test (optional): run only N queries

```bash
python -m benchmarks.vidore_tatdqa_test.sweep_eval \
  --dataset vidore/tatdqa_test \
  --collection vidore_tatdqa_test \
  --prefer-grpc \
  --mode single_full \
  --torch-dtype auto \
  --query-batch-size 32 \
  --top-k 10 \
  --max-queries 50 \
  --out-dir results/sweeps
```


