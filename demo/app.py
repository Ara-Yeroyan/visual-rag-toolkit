"""Main entry point for the Visual RAG Toolkit demo application."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root (works both locally and in Docker)
_app_dir = Path(__file__).parent
_repo_root = _app_dir.parent
if (_repo_root / ".env").exists():
    load_dotenv(_repo_root / ".env")
if (_app_dir / ".env").exists():
    load_dotenv(_app_dir / ".env")

import streamlit as st

st.set_page_config(
    page_title="Visual RAG Toolkit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from demo.ui.header import render_header
from demo.ui.sidebar import render_sidebar
from demo.ui.upload import render_upload_tab
from demo.ui.playground import render_playground_tab
from demo.ui.benchmark import render_benchmark_tab


def main():
    render_header()
    render_sidebar()

    tab_upload, tab_playground, tab_benchmark = st.tabs(
        ["📤 Upload", "🎮 Playground", "📊 Benchmarking"]
    )

    with tab_upload:
        render_upload_tab()

    with tab_playground:
        render_playground_tab()

    with tab_benchmark:
        render_benchmark_tab()


if __name__ == "__main__":
    main()
