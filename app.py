"""QuanAra — Quantitative AI for Aviation & Tourism. Streamlit entry point.

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from quanara.ui import brand

st.set_page_config(page_title=f"{brand.NAME} · {brand.TAGLINE}", page_icon=str(brand.FAVICON), layout="wide", initial_sidebar_state="auto")

from quanara.ui.common import inject_css  # noqa: E402
from quanara.ui.pages import SECTIONS  # noqa: E402

inject_css()
st.logo(str(brand.LOGO), size="large", icon_image=str(brand.MARK))

nav = st.navigation(SECTIONS, position="sidebar", expanded=True)
nav.run()

with st.sidebar:
    st.markdown(f'<div class="q-globe">{brand.globe_svg()}</div><div class="q-side-foot"><b>Quan<i>Ara</i></b>'
                f'<span>{brand.COMPETITION}</span></div>', unsafe_allow_html=True)
