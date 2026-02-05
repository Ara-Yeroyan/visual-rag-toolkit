"""Configuration constants for the demo app."""

AVAILABLE_MODELS = [
    "vidore/colpali-v1.3",
    "vidore/colSmol-500M",
]

BENCHMARK_DATASETS = [
    "vidore/esg_reports_v2",
    "vidore/biomedical_lectures_v2",
    "vidore/economics_reports_v2",
]

DATASET_STATS = {
    "vidore/esg_reports_v2": {"docs": 1538, "queries": 228},
    "vidore/biomedical_lectures_v2": {"docs": 1016, "queries": 640},
    "vidore/economics_reports_v2": {"docs": 452, "queries": 232},
}

RETRIEVAL_MODES = [
    "single_full",
    "single_tiles",
    "single_global",
    "two_stage",
    "three_stage",
]

STAGE1_MODES = [
    "tokens_vs_tiles",
    "tokens_vs_experimental",
    "pooled_query_vs_tiles",
    "pooled_query_vs_experimental",
    "pooled_query_vs_global",
]
