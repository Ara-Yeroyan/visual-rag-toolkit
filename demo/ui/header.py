"""Header component."""

import streamlit as st


def render_header():
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 15px 0;">
        <h1 style="
            font-family: 'Georgia', serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: #1a1a2e;
            letter-spacing: 3px;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        ">
            🔬 Visual RAG Toolkit
        </h1>
        <p style="
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 0.95rem;
            color: #666;
            margin-top: 5px;
            letter-spacing: 1px;
        ">
            SIGIR 2026 Demo - Multi-Vector Visual Document Retrieval
        </p>
    </div>
    """, unsafe_allow_html=True)
