"""Navigation registry — the nine sections in the required order, plus the Judges' Demo."""
from __future__ import annotations

import streamlit as st

from . import copilot_page, data_intel, demo, eda, evaluation_page, features_page, methodology, overview, simulator, training_page

PAGES = {
    "home": st.Page(overview.render, title="1 · Welcome & Overview", icon=":material/flight:", url_path="welcome", default=True),
    "data": st.Page(data_intel.render, title="2 · Data Intelligence", icon=":material/database:", url_path="data-intelligence"),
    "features": st.Page(features_page.render, title="3 · Feature Engineering", icon=":material/tune:", url_path="feature-engineering"),
    "eda": st.Page(eda.render, title="4 · Correlation & Visualization Lab", icon=":material/scatter_plot:", url_path="correlation-lab"),
    "train": st.Page(training_page.render, title="5 · ML Training Studio", icon=":material/model_training:", url_path="training-studio"),
    "eval": st.Page(evaluation_page.render, title="6 · Model Evaluation", icon=":material/monitoring:", url_path="model-evaluation"),
    "sim": st.Page(simulator.render, title="7 · What-if Simulator", icon=":material/flight_takeoff:", url_path="what-if-simulator"),
    "copilot": st.Page(copilot_page.render, title="8 · AI Copilot", icon=":material/hub:", url_path="ai-copilot"),
    "method": st.Page(methodology.render, title="9 · Methodology & Limitations", icon=":material/menu_book:", url_path="methodology"),
    "demo": st.Page(demo.render, title="Judges' Demo", icon=":material/verified:", url_path="judges-demo"),
}

SECTIONS = {
    "Scientific workflow": [PAGES[k] for k in ("home", "data", "features", "eda", "train", "eval", "sim", "copilot", "method")],
    "Guided tour": [PAGES["demo"]],
}
