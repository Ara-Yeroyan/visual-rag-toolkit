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
  --stage1-mode tokens_vs_standard_pooling \
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

---

# ViDoRe v2 BEIR datasets (Qdrant) — commands

This section indexes the **3 ViDoRe v2** datasets used in the demo UI:

- `vidore/esg_reports_v2`
- `vidore/biomedical_lectures_v2`
- `vidore/economics_reports_v2`

We use **`vidore/colqwen2.5-v0.2`**, **no cropping**, **no Cloudinary**, **gRPC**, and **float32** for both compute and stored vectors.

## Environment

```bash
export QDRANT_URL="https://YOUR_QDRANT_HOST:6333"
export QDRANT_API_KEY="YOUR_KEY"  # optional for local Qdrant
```

Optional (recommended on machines with small disks):

```bash
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_CACHE="$PWD/.cache/huggingface"
```

## Index only (no evaluation)

```bash
python -m benchmarks.vidore_beir_qdrant.run_qdrant_beir \
  --datasets \
    vidore/esg_reports_v2 \
    vidore/biomedical_lectures_v2 \
    vidore/economics_reports_v2 \
  --collection vidore_v2__colqwen25_fp32 \
  --model vidore/colqwen2.5-v0.2 \
  --index \
  --recreate \
  --indexing-threshold 0 \
  --full-scan-threshold 0 \
  --prefer-grpc \
  --torch-dtype float32 \
  --qdrant-vector-dtype float32 \
  --batch-size 1 \
  --upload-batch-size 4 \
  --upload-workers 0 \
  --no-cloudinary \
  --no-eval
```

Notes:
- **`--batch-size 1`** is the safest starting point on Apple Silicon (MPS). Increase cautiously if stable.
- This does **not** enable cropping (we do **not** pass `--crop-empty`).

## Evaluate later (optional)

Single-stage full MaxSim:

```bash
python -m benchmarks.vidore_beir_qdrant.run_qdrant_beir \
  --datasets \
    vidore/esg_reports_v2 \
    vidore/biomedical_lectures_v2 \
    vidore/economics_reports_v2 \
  --collection vidore_v2__colqwen25_fp32 \
  --model vidore/colqwen2.5-v0.2 \
  --prefer-grpc \
  --torch-dtype float32 \
  --qdrant-vector-dtype float32 \
  --mode single_full \
  --top-k 100 \
  --evaluation-scope per_dataset
```

Two-stage (prefetch + rerank):

```bash
python -m benchmarks.vidore_beir_qdrant.run_qdrant_beir \
  --datasets \
    vidore/esg_reports_v2 \
    vidore/biomedical_lectures_v2 \
    vidore/economics_reports_v2 \
  --collection vidore_v2__colqwen25_fp32 \
  --model vidore/colqwen2.5-v0.2 \
  --prefer-grpc \
  --torch-dtype float32 \
  --qdrant-vector-dtype float32 \
  --mode two_stage \
  --stage1-mode tokens_vs_experimental_pooling \
  --prefetch-k 200 \
  --top-k 100 \
  --evaluation-scope per_dataset
```


