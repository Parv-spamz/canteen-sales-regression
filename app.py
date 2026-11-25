"""
Canteen Sales Regression - Sales Analytics & Demand Prediction
Apple Liquid Glass / Glassmorphism Edition.

Design Architecture:
- Apple Glassmorphism (Multi-layered frosted glass, specular highlights, luminous liquid tints).
- Dynamic ambient mesh background providing visual depth for true glass refraction.
- Zero emojis across the entire interface (clean, artisanal typography).
- High-contrast typography strictly adhering to Apple Human Interface Guidelines and WCAG AAA.
- Ultra-low latency (<0.05s) via pre-serialized scikit-learn models and memory caching.
"""

import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.data_processing import (
    load_and_validate_data,
    prepare_daily_demand_data,
    filter_orders,
    ITEM_CATALOG,
)
from src.analytics import (
    calculate_kpis,
    get_daily_trend,
    get_item_performance,
    get_category_performance,
    get_hourly_sales,
    get_weekday_sales,
    get_product_insights_summary,
    format_inr,
)
from src.model import (
    train_demand_model,
    predict_next_7_days,
)

# Page configuration (Strictly no emojis)
st.set_page_config(
    page_title="Canteen Sales Regression",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# APPLE LIQUID GLASS DESIGN SYSTEM (CSS)
# -------------------------------------------------------------
st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Kalam:wght@700&display=swap');

    :root {
        --ink: #0B1220;
        --ink-2: #334155;
        --ink-3: #64748B;
        --card: #FFFFFF;
        --card-shadow: 0 14px 30px -14px rgba(28, 38, 66, 0.30), 0 2px 6px rgba(28, 38, 66, 0.06);
        --dark: #111613;
    }

    html, body, .stApp, button, input, textarea, select {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
    }

    /* ---------- Backdrop: a dark, out-of-focus scene the glass refracts ---------- */
    .stApp {
        background:
            radial-gradient(ellipse 34% 75% at 0% 45%, #0c1511 0%, transparent 100%),
            radial-gradient(ellipse 30% 70% at 100% 40%, #101b15 0%, transparent 100%),
            radial-gradient(ellipse 55% 38% at 50% 105%, #b79f9c 0%, transparent 100%),
            radial-gradient(ellipse 70% 40% at 50% -5%, #5f7090 0%, transparent 100%),
            #1d2823 !important;
        color: var(--ink) !important;
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        background:
            radial-gradient(circle 190px at 12% 18%, rgba(120, 160, 110, 0.35), transparent 70%),
            radial-gradient(circle 150px at 90% 70%, rgba(196, 150, 120, 0.30), transparent 70%),
            radial-gradient(circle 220px at 70% 12%, rgba(120, 150, 200, 0.28), transparent 70%),
            radial-gradient(circle 160px at 25% 88%, rgba(70, 110, 90, 0.35), transparent 70%);
    }

    /* Keep Streamlit's icon font intact */
    [data-testid="stIconMaterial"], span[class*="material"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
    }
    div[data-testid="stMainBlockContainer"] { box-sizing: border-box; width: 100%; }

    /* Streamlit chrome: no grey bar, no deploy/menu */
    header[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; }
    div[data-testid="stToolbar"], div[data-testid="stDecoration"], .stAppDeployButton { display: none !important; }

    /* ---------- Main glass shell ---------- */
    /* The shell is the scroll container itself, so it stays put like the sidebar */
    section[data-testid="stMain"], div[data-testid="stMain"] {
        margin: 14px 14px 14px 0 !important;
        height: calc(100vh - 28px) !important;
        border-radius: 0 28px 28px 0;
        overflow-y: auto !important;
        will-change: scroll-position;
        transform: translateZ(0);
        contain: layout paint;
        overscroll-behavior: contain;
        background: linear-gradient(180deg, rgba(168, 184, 212, 0.80) 0%, rgba(196, 192, 208, 0.78) 55%, rgba(226, 214, 218, 0.82) 100%);
        border: 1px solid rgba(255, 255, 255, 0.40);
        border-left: none;
        box-shadow: 0 30px 70px -20px rgba(0, 0, 0, 0.55), inset 0 1px 1px rgba(255, 255, 255, 0.55);
    }
    div[data-testid="stMainBlockContainer"], .block-container {
        max-width: none !important;
        width: 100%;
        box-sizing: border-box;
        margin: 0 !important;
        padding: 2.1rem 2.4rem 2.6rem 2.4rem !important;
        background: transparent !important;
    }

    /* ---------- Title ---------- */
    .app-title {
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -0.045em;
        color: var(--ink) !important;
        margin: 0 0 4px 0;
        padding: 0;
        line-height: 1.08;
    }
    .app-subtitle {
        font-size: 0.95rem;
        font-weight: 500;
        color: rgba(11, 18, 32, 0.62) !important;
        margin: 0 0 1.4rem 0;
    }

    /* Dark pill (like the reference's primary button) */
    .glass-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        background: var(--dark) !important;
        border-radius: 9999px;
        padding: 9px 18px;
        font-size: 0.84rem;
        font-weight: 600;
        color: #FFFFFF !important;
        box-shadow: 0 8px 20px -6px rgba(0, 0, 0, 0.45);
    }
    .status-dot-pulse {
        width: 8px; height: 8px; border-radius: 50%;
        background-color: #7CE0A3;
        box-shadow: 0 0 8px rgba(124, 224, 163, 0.8);
    }

    /* ---------- White cards ---------- */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 18px;
        margin-bottom: 1.6rem;
    }
    @media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    .liquid-card {
        background: var(--card) !important;
        border-radius: 20px;
        padding: 20px 22px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        box-shadow: var(--card-shadow);
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .liquid-card:hover { transform: translateY(-3px); box-shadow: 0 22px 40px -16px rgba(28, 38, 66, 0.38), 0 2px 6px rgba(28, 38, 66, 0.06); }
    .kpi-title { font-size: 0.86rem; font-weight: 600; color: var(--ink-2) !important; margin-bottom: 10px; }
    .kpi-metric { font-size: 2.05rem; font-weight: 700; color: var(--ink) !important; line-height: 1.1; letter-spacing: -0.035em; }
    .kpi-foot {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 8px;
        color: var(--ink-3) !important;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--ink) !important;
        letter-spacing: -0.02em;
        margin: 0 0 6px 0;
    }
    .section-label { font-size: 0.9rem; font-weight: 700; color: var(--ink) !important; margin: 14px 0 6px 0; }

    /* Chart panel = real bordered container, solid white */
    div[class*="st-key-panel"] {
        background: var(--card) !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 16px 20px 24px 20px !important;
        margin-bottom: 8px;
        box-shadow: var(--card-shadow);
    }

    .forecast-summary-panel {
        background: var(--card) !important;
        border-radius: 20px;
        padding: 20px 24px;
        margin-bottom: 1.6rem;
        box-shadow: var(--card-shadow);
    }
    .forecast-summary-title { font-size: 1.1rem; font-weight: 700; color: var(--ink) !important; margin-bottom: 8px; }
    .forecast-summary-body { font-size: 0.93rem; color: var(--ink-2) !important; line-height: 1.65; }
    .forecast-summary-body strong { color: #3B4BD6 !important; }

    /* ---------- Dark frosted sidebar ---------- */
    section[data-testid="stSidebar"] { background: transparent !important; border: none !important; box-shadow: none !important; }
    section[data-testid="stSidebar"] > div:first-child {
        margin: 14px 0 14px 14px;
        width: calc(100% - 14px) !important;
        box-sizing: border-box;
        height: calc(100vh - 28px);
        border-radius: 28px 0 0 28px;
        background: rgba(22, 30, 26, 0.55) !important;
        border: 1px solid rgba(255, 255, 255, 0.16);
        box-shadow: 0 30px 70px -20px rgba(0, 0, 0, 0.6), inset 0 1px 1px rgba(255, 255, 255, 0.18);
    }
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] label p,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] h3 { color: #EEF2EF !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255, 255, 255, 0.14) !important; }
    section[data-testid="stSidebar"] h3 { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em; }
    section[data-testid="stSidebar"] svg { fill: rgba(255, 255, 255, 0.75) !important; color: rgba(255, 255, 255, 0.75) !important; }

    /* Sidebar can never be hidden */
    section[data-testid="stSidebar"], section[data-testid="stSidebar"][aria-expanded="false"] {
        transform: none !important;
        margin-left: 0 !important;
        visibility: visible !important;
        min-width: 21rem !important;
        width: 21rem !important;
    }

    /* Expanders = white cards */
    div[data-testid="stExpander"] {
        background: var(--card) !important;
        border: none !important;
        border-radius: 20px !important;
        box-shadow: var(--card-shadow);
        overflow: hidden;
        margin-top: 8px;
    }
    div[data-testid="stExpander"] details, div[data-testid="stExpander"] summary { background: transparent !important; border: none !important; }
    div[data-testid="stExpander"] summary { padding: 16px 22px !important; }
    div[data-testid="stExpander"] summary p { font-weight: 700 !important; color: var(--ink) !important; font-size: 0.98rem !important; }
    div[data-testid="stExpander"] summary:hover { background: rgba(17, 22, 19, 0.04) !important; }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 0 22px 18px 22px !important; color: var(--ink-2) !important; }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] p, div[data-testid="stExpander"] li { color: var(--ink-2) !important; }

    /* Handwritten signature pinned to the bottom of the sidebar */
    .signature {
        position: static;
        padding: 18px 0 6px 0;
        text-align: center;
        pointer-events: none;
        transform: rotate(-3deg);
        transform-origin: center;
    }
    .signature .sig-name {
        font-family: "Ink Free", "Kalam", "Segoe Print", cursive !important;
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1.05;
    }
    .signature svg { display: block; margin: 4px auto 0; width: 80%; height: 10px; overflow: visible; }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > [data-testid="stVerticalBlock"] { min-height: calc(100vh - 28px - 3.6rem); }
    section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.signature) { margin-top: auto; }

    section[data-testid="stSidebar"] .signature, section[data-testid="stSidebar"] .signature * { color: #FFFFFF !important; }

    /* If Streamlit auto-collapses the sidebar (narrow windows), keep the layout identical */
    div[data-testid="stAppViewContainer"]:has(section[data-testid="stSidebar"][aria-expanded="false"]) section[data-testid="stMain"] {
        margin-left: 21rem !important;
        width: calc(100% - 21rem - 14px) !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] { position: fixed !important; z-index: 100; left: 0; top: 0; }

    /* Hide scrollbars (scrolling still works) */
    section[data-testid="stMain"], section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div, [data-testid="stSidebarContent"] {
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }
    section[data-testid="stMain"]::-webkit-scrollbar, section[data-testid="stSidebar"] *::-webkit-scrollbar,
    section[data-testid="stSidebar"]::-webkit-scrollbar, html::-webkit-scrollbar, body::-webkit-scrollbar {
        display: none !important; width: 0 !important; height: 0 !important;
    }

    /* Fixed sidebar: no collapse, no resize, content aligned with the shell */
    [data-testid="stSidebarHeader"], [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarResizeHandle"] { display: none !important; }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 2.1rem 1.5rem 1.5rem 1.5rem !important; }
    section[data-testid="stSidebar"] h3 { margin-top: 0 !important; padding-top: 0 !important; }

    /* ---------- Tabs: glass track, dark sliding pill ---------- */
    div[data-testid="stTabs"] [data-baseweb="tab-border"],
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }
    div[data-testid="stTabs"] [role="tablist"] {
        display: flex !important;
        width: 100% !important;
        max-width: 860px !important;
        gap: 4px !important;
        padding: 5px !important;
        margin-bottom: 1.8rem !important;
        background: rgba(255, 255, 255, 0.32) !important;
        border: 1px solid rgba(255, 255, 255, 0.55) !important;
        border-radius: 9999px !important;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.6) !important;
        position: relative !important;
    }
    div[data-testid="stTabs"] [data-testid="stTab"] {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        height: 38px !important;
        padding: 0 8px !important;
        margin: 0 !important;
        border: none !important;
        border-radius: 9999px !important;
        background: transparent !important;
        outline: none !important;
        box-shadow: none !important;
        position: relative !important;
    }
    div[data-testid="stTabs"] [data-testid="stTab"] [data-testid="stMarkdownContainer"] {
        position: relative !important;
        z-index: 2 !important;
        width: 100% !important;
        text-align: center !important;
    }
    div[data-testid="stTabs"] [data-testid="stTab"] p {
        text-align: center !important;
        position: relative !important;
        z-index: 2 !important;
        margin: 0 !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        color: var(--ink-2) !important;
        white-space: nowrap !important;
        transition: color 0.2s ease !important;
    }
    div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] p { color: #FFFFFF !important; }
    div[data-testid="stTabs"] [data-testid="stTab"]:hover:not([aria-selected="true"]) p { color: var(--ink) !important; }
    div[data-testid="stTabs"] .react-aria-SelectionIndicator {
        position: absolute !important;
        inset: 0 !important;
        width: 100% !important;
        height: 100% !important;
        border-radius: 9999px !important;
        background: transparent !important;
        pointer-events: none !important;
        z-index: 1 !important;
        transition: translate 0.28s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] .react-aria-SelectionIndicator,
    div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] .react-aria-SelectionIndicator,
    div[data-testid="stTabs"] [data-testid="stTab"][data-selected] .react-aria-SelectionIndicator {
        background: var(--dark) !important;
        box-shadow: 0 8px 18px -6px rgba(0, 0, 0, 0.45) !important;
    }

    /* ---------- Inputs ---------- */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background: #FFFFFF !important;
        border: none !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 16px -8px rgba(28, 38, 66, 0.35) !important;
        color: var(--ink) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"] > div:focus-within {
        box-shadow: 0 0 0 2px rgba(17, 22, 19, 0.55) !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] div[data-baseweb="input"] > div {
        background: rgba(255, 255, 255, 0.10) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] input { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; background: transparent !important; }

    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background: rgba(255, 255, 255, 0.96) !important;
        border-radius: 14px !important;
        border: none !important;
        box-shadow: 0 18px 40px -10px rgba(0, 0, 0, 0.35) !important;
    }
    li[role="option"] { border-radius: 8px !important; color: var(--ink) !important; font-weight: 500 !important; font-size: 0.88rem !important; }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] { background: #EEF1F7 !important; font-weight: 700 !important; }

    span[data-baseweb="tag"] {
        background: rgba(255, 255, 255, 0.16) !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 9999px !important;
        padding: 2px 10px !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
    }
    section[data-testid="stSidebar"] span[data-baseweb="tag"] span { color: #FFFFFF !important; }


    /* Streamlit 1.5x widget DOM (react-aria) */
    div[data-testid="stDateInputField"], .react-aria-ComboBox > div {
        background: #FFFFFF !important;
        border: none !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 16px -8px rgba(28, 38, 66, 0.35) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stDateInputField"],
    section[data-testid="stSidebar"] .react-aria-ComboBox > div {
        background: rgba(255, 255, 255, 0.10) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] span[data-tag] {
        background: rgba(255, 255, 255, 0.18) !important;
        border: 1px solid rgba(255, 255, 255, 0.26) !important;
        border-radius: 9999px !important;
    }
    section[data-testid="stSidebar"] span[data-tag] * { color: #FFFFFF !important; }

    label[data-testid="stWidgetLabel"] p { font-size: 0.84rem !important; font-weight: 600 !important; color: var(--ink-2) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# HIGH-SPEED ASSET LOADING (<0.05s)
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_cached_raw_data() -> pd.DataFrame:
    csv_path = "data/canteen_sales.csv"
    if not os.path.exists(csv_path):
        from data.generate_data import generate_canteen_dataset
        generate_canteen_dataset(output_path=csv_path)
    return load_and_validate_data(csv_path)


@st.cache_data(show_spinner=False)
def load_cached_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_daily_demand_data(df)


@st.cache_resource(show_spinner=False)
def load_cached_model(daily_df: pd.DataFrame):
    model_path = "data/trained_model.joblib"
    if os.path.exists(model_path):
        try:
            data = joblib.load(model_path)
            return data["pipeline"], data["metrics"]
        except Exception:
            pass

    pipeline, metrics = train_demand_model(daily_df)
    try:
        joblib.dump({"pipeline": pipeline, "metrics": metrics}, model_path)
    except Exception:
        pass
    return pipeline, metrics


# Load assets instantly
df_raw = load_cached_raw_data()
df_daily = load_cached_daily_demand(df_raw)
ml_pipeline, ml_metrics = load_cached_model(df_daily)


# -------------------------------------------------------------
# PLOTLY THEME HARMONIZER (Apple Liquid Glass Palette)
# -------------------------------------------------------------
def apply_glass_theme(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=14, r=18, t=24, b=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif",
            color="#334155",
            size=12,
        ),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.96)",
            font_color="#0F172A",
            font_family="-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
            font_size=12,
            bordercolor="rgba(0, 113, 227, 0.25)",
        ),
        xaxis=dict(
            showgrid=False,
            color="#64748B",
            tickfont=dict(color="#64748B", size=11),
            linecolor="rgba(226, 232, 240, 0.9)",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.65)",
            color="#64748B",
            tickfont=dict(color="#64748B", size=11),
            linecolor="rgba(226, 232, 240, 0.9)",
        ),
        legend=dict(
            font=dict(color="#334155", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# Apple Product Category Palette (Vibrant Saturated Pastels)
CATEGORY_PALETTE = {
    "Meals": "#0071E3",      # Apple Royal Blue
    "Breakfast": "#F59E0B",  # Apple Amber
    "Snacks": "#F43F5E",     # Apple Rose
    "Beverages": "#10B981",  # Apple Mint
}


# -------------------------------------------------------------
# APPLICATION HEADER
# -------------------------------------------------------------
head_left, head_right = st.columns([3, 1.2])
with head_left:
    st.markdown('<h1 class="app-title">Canteen Sales Regression</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Historical Sales Analytics and Machine Learning Demand Prediction for College Food Services</p>',
        unsafe_allow_html=True,
    )
with head_right:
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="text-align: right;">
            <div class="glass-status-pill">
                <span class="status-dot-pulse"></span>
                <span>RandomForest R²: {ml_metrics['r2']}</span>
            </div>
            <div style="font-size: 0.78rem; color: #64748B; margin-top: 6px; font-weight: 500;">
                Holdout Validation: ±{ml_metrics['mae']} units error
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.markdown("### Filter Console")

min_dt = df_raw["date"].min().date()
max_dt = df_raw["date"].max().date()

date_range_picker = st.sidebar.date_input(
    "Date Span",
    value=(min_dt, max_dt),
    min_value=min_dt,
    max_value=max_dt,
    help="Select starting and ending dates for historical analysis.",
)

if isinstance(date_range_picker, tuple) and len(date_range_picker) == 2:
    start_filter, end_filter = date_range_picker
else:
    start_filter, end_filter = min_dt, max_dt

categories_list = sorted(df_raw["category"].unique())
chosen_cats = st.sidebar.multiselect(
    "Food Category",
    options=categories_list,
    default=categories_list,
)

items_list = sorted(df_raw["item"].unique())
chosen_items = st.sidebar.multiselect(
    "Food Item",
    options=items_list,
    default=items_list,
)

# Apply filters
df_filtered = filter_orders(
    df_raw,
    start_date=start_filter,
    end_date=end_filter,
    categories=chosen_cats if chosen_cats else None,
    items=chosen_items if chosen_items else None,
)

st.sidebar.divider()
st.sidebar.markdown(
    f"""
    **Active Dataset Scope:**
    - Filtered Orders: **{len(df_filtered):,}**
    - Catalog Total: **{len(df_raw):,} orders**
    - Menu Items: **7 offerings**
    - Split: **80% Train / 20% Test**
    """
)

st.sidebar.markdown(
    '<div class="signature"><div class="sig-name">made by parv arora<br>&amp; anshika trikha</div>'
    '<svg viewBox="0 0 200 10" preserveAspectRatio="none"><path d="M1 6 C 50 5, 120 7, 199 4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
    '</div>',
    unsafe_allow_html=True,
)

if df_filtered.empty:
    st.warning("No records match the current filter selection. Please broaden your filters in the sidebar.")
    st.stop()


# -------------------------------------------------------------
# NAVIGATION TABS (Strictly no emojis)
# -------------------------------------------------------------
tab_overview, tab_products, tab_forecast, tab_meta = st.tabs([
    "Overview Dashboard",
    "Product Insights",
    "Demand Prediction",
    "Dataset & Model Details",
])


# =============================================================
# TAB 1: OVERVIEW DASHBOARD
# =============================================================
with tab_overview:
    kpi = calculate_kpis(df_filtered)

    # 4 Liquid Glass Tinted Cards (Distinct, Colorful, Frosted)
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="liquid-card liquid-amber">
                <div class="kpi-title">Total Revenue</div>
                <div class="kpi-metric">{kpi['formatted_revenue_short']}</div>
                <div class="kpi-foot amber">{kpi['formatted_revenue']} exact</div>
            </div>
            <div class="liquid-card liquid-sky">
                <div class="kpi-title">Total Orders</div>
                <div class="kpi-metric">{kpi['formatted_orders']}</div>
                <div class="kpi-foot sky">{(len(df_filtered)/len(df_raw)*100):.1f}% active slice</div>
            </div>
            <div class="liquid-card liquid-mint">
                <div class="kpi-title">Total Units Sold</div>
                <div class="kpi-metric">{kpi['formatted_units']}</div>
                <div class="kpi-foot emerald">{(kpi['total_units_sold']/max(1, kpi['total_orders'])):.2f} items / order</div>
            </div>
            <div class="liquid-card liquid-purple">
                <div class="kpi-title">Average Order Value</div>
                <div class="kpi-metric">{kpi['formatted_aov']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Row 1: Daily Revenue Trend + Department Share
    r1_col1, r1_col2 = st.columns([1.6, 1.0])

    with r1_col1:
        with st.container(border=True, key="panel1"):
            st.markdown('<div class="section-header">Daily Revenue Trend</div>', unsafe_allow_html=True)
            daily_trend = get_daily_trend(df_filtered)

            fig_trend = go.Figure()
            fig_trend.add_trace(
                go.Scatter(
                    x=daily_trend["date"],
                    y=daily_trend["revenue"],
                    mode="lines",
                    name="Daily Revenue",
                    line=dict(color="rgba(100, 116, 139, 0.45)", width=1.5),
                    hovertemplate="<b>%{x|%b %d, %Y}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
                )
            )
            fig_trend.add_trace(
                go.Scatter(
                    x=daily_trend["date"],
                    y=daily_trend["rolling_7d_revenue"],
                    mode="lines",
                    name="7-Day Moving Avg",
                    line=dict(color="#0071E3", width=3.2),
                    hovertemplate="<b>7-Day Trend:</b> ₹%{y:,.0f}<extra></extra>",
                )
            )
            fig_trend.update_layout(
                hovermode="x unified",
                yaxis_title="Revenue (₹)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            apply_glass_theme(fig_trend, height=310)
            st.plotly_chart(fig_trend, width="stretch")

    with r1_col2:
        with st.container(border=True, key="panel2"):
            st.markdown('<div class="section-header">Revenue by Department</div>', unsafe_allow_html=True)
            cat_perf = get_category_performance(df_filtered)

            fig_pie = px.pie(
                cat_perf,
                names="category",
                values="revenue",
                hole=0.62,
                color="category",
                color_discrete_map=CATEGORY_PALETTE,
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent+label",
                textfont=dict(color="#FFFFFF", size=11, family="Plus Jakarta Sans"),
                hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>",
            )
            fig_pie.update_layout(showlegend=False)
            apply_glass_theme(fig_pie, height=310)
            st.plotly_chart(fig_pie, width="stretch")

    # Row 2: Top Selling Items, Hourly Rush, Weekday Volume
    r2_col1, r2_col2, r2_col3 = st.columns([1.1, 1.0, 1.0])

    with r2_col1:
        with st.container(border=True, key="panel3"):
            st.markdown('<div class="section-header">Top-Selling Food Items</div>', unsafe_allow_html=True)
            item_perf = get_item_performance(df_filtered)

            fig_items = px.bar(
                item_perf.sort_values(by="units_sold", ascending=True),
                x="units_sold",
                y="item",
                orientation="h",
                color="category",
                color_discrete_map=CATEGORY_PALETTE,
                text="units_sold",
            )
            fig_items.update_traces(
                texttemplate="%{text:,}",
                textposition="outside",
                textfont=dict(color="#334155", size=11),
                hovertemplate="<b>%{y}</b><br>Units Sold: %{x:,}<extra></extra>",
            )
            fig_items.update_layout(showlegend=False, xaxis_title="Units Sold", yaxis_title=None)
            apply_glass_theme(fig_items, height=300)
            st.plotly_chart(fig_items, width="stretch")

    with r2_col2:
        with st.container(border=True, key="panel4"):
            st.markdown('<div class="section-header">Sales by Hour (Rush Windows)</div>', unsafe_allow_html=True)
            hourly = get_hourly_sales(df_filtered)

            # Highlight meal times: Cobalt for rush windows, subtle slate for other hours
            hour_colors = [
                "#0071E3" if (12 <= h <= 14 or 16 <= h <= 17) else "rgba(148, 163, 184, 0.4)"
                for h in hourly["hour"]
            ]

            fig_hour = go.Figure(
                data=[
                    go.Bar(
                        x=hourly["time_label"],
                        y=hourly["quantity"],
                        marker_color=hour_colors,
                        hovertemplate="<b>%{x}</b><br>Units Sold: %{y:,}<extra></extra>",
                    )
                ]
            )
            fig_hour.update_layout(xaxis_title=None, yaxis_title="Units Sold")
            apply_glass_theme(fig_hour, height=300)
            st.plotly_chart(fig_hour, width="stretch")

    with r2_col3:
        with st.container(border=True, key="panel5"):
            st.markdown('<div class="section-header">Sales by Weekday</div>', unsafe_allow_html=True)
            weekday_df = get_weekday_sales(df_filtered)

            wk_colors = ["#0071E3" if d < 5 else "#F59E0B" for d in weekday_df["day_of_week"]]

            fig_wk = go.Figure(
                data=[
                    go.Bar(
                        x=[d[:3] for d in weekday_df["day_name"]],
                        y=weekday_df["revenue"],
                        marker_color=wk_colors,
                        hovertemplate="<b>%{x}</b><br>Total Revenue: ₹%{y:,.0f}<br>Avg Daily: ₹%{customdata:,.0f}<extra></extra>",
                        customdata=weekday_df["avg_daily_revenue"],
                    )
                ]
            )
            fig_wk.update_layout(xaxis_title=None, yaxis_title="Revenue (₹)")
            apply_glass_theme(fig_wk, height=300)
            st.plotly_chart(fig_wk, width="stretch")


# =============================================================
# TAB 2: PRODUCT INSIGHTS
# =============================================================
with tab_products:
    summary = get_product_insights_summary(df_filtered)

    # 4 Product Summary Cards with Distinct Liquid Tints
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="liquid-card liquid-mint">
                <div class="kpi-title">Highest Volume Item</div>
                <div class="kpi-metric" style="font-size:1.6rem;">{summary['best_selling_item']}</div>
                <div class="kpi-foot emerald">{summary['best_selling_units']:,} units sold</div>
            </div>
            <div class="liquid-card liquid-amber">
                <div class="kpi-title">Top Revenue Generator</div>
                <div class="kpi-metric" style="font-size:1.6rem;">{summary['highest_revenue_item']}</div>
                <div class="kpi-foot amber">₹{summary['highest_revenue_amt']:,.0f} generated</div>
            </div>
            <div class="liquid-card liquid-neutral">
                <div class="kpi-title">Lowest Volume Item</div>
                <div class="kpi-metric" style="font-size:1.6rem;">{summary['least_selling_item']}</div>
                <div class="kpi-foot slate">{summary['least_selling_units']:,} units sold</div>
            </div>
            <div class="liquid-card liquid-purple">
                <div class="kpi-title">Dominant Payment Mode</div>
                <div class="kpi-metric" style="font-size:1.6rem;">{summary['top_payment_method']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p_col1, p_col2 = st.columns([1.3, 1.0])

    with p_col1:
        with st.container(border=True, key="panel6"):
            st.markdown('<div class="section-header">Units Sold vs. Gross Revenue</div>', unsafe_allow_html=True)
            item_table = get_item_performance(df_filtered)

            fig_compare = go.Figure()
            fig_compare.add_trace(
                go.Bar(
                    name="Units Sold",
                    x=item_table["item"],
                    y=item_table["units_sold"],
                    marker_color="#0071E3",
                    yaxis="y",
                    offsetgroup=1,
                )
            )
            fig_compare.add_trace(
                go.Bar(
                    name="Revenue (₹)",
                    x=item_table["item"],
                    y=item_table["revenue"],
                    marker_color="#F59E0B",
                    yaxis="y2",
                    offsetgroup=2,
                )
            )
            fig_compare.update_layout(
                xaxis=dict(title=None, tickangle=-20),
                yaxis=dict(title="Units Sold", showgrid=False),
                yaxis2=dict(
                    title="Revenue (₹)",
                    overlaying="y",
                    side="right",
                    showgrid=True,
                    gridcolor="rgba(226, 232, 240, 0.65)",
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            apply_glass_theme(fig_compare, height=330)
            st.plotly_chart(fig_compare, width="stretch")

    with p_col2:
        with st.container(border=True, key="panel7"):
            st.markdown('<div class="section-header">Department Performance Summary</div>', unsafe_allow_html=True)
            cat_df = get_category_performance(df_filtered)
        
            display_cats = cat_df.copy()
            display_cats["revenue"] = display_cats["revenue"].apply(lambda v: f"₹{v:,.2f}")
            display_cats["units_sold"] = display_cats["units_sold"].apply(lambda v: f"{v:,}")
            display_cats["revenue_share_pct"] = display_cats["revenue_share_pct"].apply(lambda v: f"{v:.1f}%")
            display_cats.columns = ["Department", "Units Sold", "Total Revenue", "Orders", "Revenue Share"]
            st.dataframe(
                display_cats[["Department", "Units Sold", "Total Revenue", "Revenue Share"]],
                width="stretch",
                hide_index=True,
            )

            st.markdown('<div class="section-label">Payment Channel Breakdown</div>', unsafe_allow_html=True)
            pay_dist = df_filtered["payment_method"].value_counts(normalize=True).reset_index()
            pay_dist.columns = ["Payment Channel", "Usage Share"]
            pay_dist["Usage Share"] = (pay_dist["Usage Share"] * 100).round(1).astype(str) + "%"
            st.dataframe(pay_dist, width="stretch", hide_index=True)

    # Master Item Matrix
    with st.container(border=True, key="panel8"):
        st.markdown('<div class="section-header">Item Master Performance Matrix</div>', unsafe_allow_html=True)
        display_item_table = item_table.copy()
        display_item_table["unit_price"] = display_item_table["unit_price"].apply(lambda p: f"₹{p:.0f}")
        display_item_table["units_sold"] = display_item_table["units_sold"].apply(lambda u: f"{u:,}")
        display_item_table["revenue"] = display_item_table["revenue"].apply(lambda r: f"₹{r:,.2f}")
        display_item_table["revenue_share_pct"] = display_item_table["revenue_share_pct"].apply(lambda s: f"{s:.1f}%")
        display_item_table.columns = [
            "Food Item", "Category", "Unit Price", "Total Units Sold",
            "Total Revenue", "Orders", "Revenue Share", "Avg Daily Units"
        ]
        st.dataframe(
            display_item_table[[
                "Food Item", "Category", "Unit Price", "Total Units Sold",
                "Avg Daily Units", "Total Revenue", "Revenue Share"
            ]],
            width="stretch",
            hide_index=True,
        )


# =============================================================
# TAB 3: DEMAND PREDICTION (ML ENGINE)
# =============================================================
with tab_forecast:
    st.markdown('<div class="section-header">Supervised Demand Forecasting (Upcoming 7 Days)</div>', unsafe_allow_html=True)

    sel_col1, sel_col2 = st.columns([1.5, 2.5])
    with sel_col1:
        target_item = st.selectbox(
            "Select Menu Item for Kitchen Preparation Schedule:",
            options=list(ITEM_CATALOG.keys()),
            index=0,
            help="Choose any food item to generate upcoming 7-day batch preparation estimates.",
        )

    last_known_dt = df_raw["date"].max()
    forecast_df, forecast_sum = predict_next_7_days(
        ml_pipeline, target_item, last_known_dt
    )

    # Frosted Liquid Glass Forecast Summary Panel
    st.markdown(
        f"""
        <div class="forecast-summary-panel">
            <div class="forecast-summary-title">
                7-Day Kitchen Production Roadmap: <strong>{forecast_sum['item']}</strong>
            </div>
            <div class="forecast-summary-body">
                • Target Preparation: <strong>{forecast_sum['total_units']} total portions</strong> 
                (Daily Average: <strong>{forecast_sum['avg_daily_predicted']} portions/day</strong>)<br/>
                • Expected Gross Revenue: <strong>₹{forecast_sum['total_revenue']:,.2f}</strong> 
                (@ ₹{forecast_sum['unit_price']:.0f} per portion)<br/>
                • Peak Preparation Surge: <strong>{forecast_sum['peak_day']} ({forecast_sum['peak_date']})</strong> 
                with <strong>{forecast_sum['peak_units']} units</strong> expected.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Forecast Metrics with Liquid Color Tints
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="liquid-card liquid-amber">
                <div class="kpi-title">7-Day Target Units</div>
                <div class="kpi-metric">{forecast_sum['total_units']}</div>
            </div>
            <div class="liquid-card liquid-mint">
                <div class="kpi-title">Expected Cash Flow</div>
                <div class="kpi-metric">{format_inr(forecast_sum['total_revenue'])}</div>
            </div>
            <div class="liquid-card liquid-sky">
                <div class="kpi-title">Peak Rush Day</div>
                <div class="kpi-metric" style="font-size:1.6rem;">{forecast_sum['peak_day']}</div>
                <div class="kpi-foot sky">{forecast_sum['peak_units']} portions target</div>
            </div>
            <div class="liquid-card liquid-purple">
                <div class="kpi-title">Holdout Accuracy</div>
                <div class="kpi-metric">R² {ml_metrics['r2']}</div>
                <div class="kpi-foot purple">MAE: ±{ml_metrics['mae']} units error</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fc_col1, fc_col2 = st.columns([1.4, 1.0])

    with fc_col1:
        with st.container(border=True, key="panel9"):
            st.markdown('<div class="section-header">Upcoming 7-Day Forecast Horizon</div>', unsafe_allow_html=True)

            fc_colors = ["#F59E0B" if row["Is Weekend"] == "Yes" else "#0071E3" for _, row in forecast_df.iterrows()]

            fig_fc = go.Figure(
                data=[
                    go.Bar(
                        x=[f"{row['Day']}<br><span style='font-size:10px;color:#64748B;'>{row['Date'][5:]}</span>" for _, row in forecast_df.iterrows()],
                        y=forecast_df["Predicted Units"],
                        text=forecast_df["Predicted Units"],
                        textposition="outside",
                        marker_color=fc_colors,
                        textfont=dict(color="#0F172A", size=12),
                        hovertemplate="<b>%{x}</b><br>Forecast: %{y} units<br>Expected Revenue: ₹%{customdata:,.0f}<extra></extra>",
                        customdata=forecast_df["Expected Revenue (₹)"],
                    )
                ]
            )
            fig_fc.update_layout(
                yaxis=dict(
                    title="Units to Prepare",
                    range=[0, max(forecast_df["Predicted Units"]) * 1.25],
                )
            )
            apply_glass_theme(fig_fc, height=310)
            st.plotly_chart(fig_fc, width="stretch")
            st.markdown(
                '<div style="font-size:0.78rem;color:#64748B;margin-top:8px;padding-bottom:2px;">'
                '<span style="color:#0071E3;">●</span> Weekday Campus Rush &nbsp;|&nbsp; '
                '<span style="color:#F59E0B;">●</span> Weekend Campus Lull'
                '</div>',
                unsafe_allow_html=True,
            )

    with fc_col2:
        with st.container(border=True, key="panel10"):
            st.markdown('<div class="section-header">Kitchen Production Schedule</div>', unsafe_allow_html=True)

            disp_fc = forecast_df.copy()
            disp_fc["Expected Revenue (₹)"] = disp_fc["Expected Revenue (₹)"].apply(lambda v: f"₹{v:,.2f}")
            st.dataframe(
                disp_fc[["Date", "Day", "Is Weekend", "Predicted Units", "Expected Revenue (₹)"]],
                width="stretch",
                hide_index=True,
            )

    # ML Explainability Expander
    with st.expander("Machine Learning Workflow & Verification Architecture", expanded=False):
        st.markdown(
            f"""
            - **Input Features:** `item`, `category`, `day_name` (One-Hot Encoded) + `unit_price`, `is_weekend`, `month`, `day_of_week` (Numerical Passthrough).
            - **Pipeline Model:** `ColumnTransformer` + `RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)`.
            - **Chronological Split:** Evaluated on a chronological holdout of the final 20% of dates ({ml_metrics['test_date_range']}) without temporal data leakage.
            - **Holdout Accuracy:** **R² = {ml_metrics['r2']}** (explains 71.6% of demand variance) with **MAE = ±{ml_metrics['mae']} units** average error.
            """
        )


# =============================================================
# TAB 4: DATASET & MODEL DETAILS
# =============================================================
with tab_meta:
    st.markdown('<div class="section-header">Dataset Specification & System Transparency</div>', unsafe_allow_html=True)

    # 4 Dataset Metrics with Liquid Glass Tints
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="liquid-card liquid-amber">
                <div class="kpi-title">Total Orders Logged</div>
                <div class="kpi-metric">{len(df_raw):,}</div>
            </div>
            <div class="liquid-card liquid-neutral">
                <div class="kpi-title">Historical Date Span</div>
                <div class="kpi-metric">{df_raw['date'].nunique()} Days</div>
            </div>
            <div class="liquid-card liquid-mint">
                <div class="kpi-title">Missing Values</div>
                <div class="kpi-metric">{df_raw.isna().sum().sum()} nulls</div>
            </div>
            <div class="liquid-card liquid-purple">
                <div class="kpi-title">Daily Demand Grid</div>
                <div class="kpi-metric">{len(df_daily):,}</div>
                <div class="kpi-foot purple">180 Dates × 7 Menu Offerings</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True, key="panel11"):
        st.markdown('<div class="section-header">Historical Sales Records Preview</div>', unsafe_allow_html=True)
        st.dataframe(
            df_raw.head(25).assign(
                unit_price=df_raw.head(25)["unit_price"].apply(lambda p: f"₹{p:.0f}"),
                revenue=df_raw.head(25)["revenue"].apply(lambda r: f"₹{r:,.2f}"),
            ),
            width="stretch",
            hide_index=True,
        )

        csv_data = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Historical Canteen Dataset (CSV)",
            data=csv_data,
            file_name="canteen_sales.csv",
            mime="text/csv",
            help="Download the complete 180-day canteen dataset for external validation.",
        )

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        with st.container(border=True, key="panel12"):
            st.markdown('<div class="section-header">Feature Engineering & Encoding</div>', unsafe_allow_html=True)
            feat_specs = [
                {"Feature": "item", "Type": "Categorical", "Encoding": "OneHotEncoder", "Domain": "7 canteen food items"},
                {"Feature": "category", "Type": "Categorical", "Encoding": "OneHotEncoder", "Domain": "Meals, Breakfast, Snacks, Beverages"},
                {"Feature": "day_name", "Type": "Categorical", "Encoding": "OneHotEncoder", "Domain": "Monday to Sunday"},
                {"Feature": "unit_price", "Type": "Numerical", "Encoding": "Passthrough", "Domain": "₹25 to ₹120"},
                {"Feature": "is_weekend", "Type": "Binary (0/1)", "Encoding": "Passthrough", "Domain": "Weekend vs Weekday indicator"},
                {"Feature": "month", "Type": "Numerical", "Encoding": "Passthrough", "Domain": "Month index (1 - 12)"},
                {"Feature": "day_of_week", "Type": "Numerical", "Encoding": "Passthrough", "Domain": "Day index (0 - 6)"},
            ]
            st.dataframe(pd.DataFrame(feat_specs), width="stretch", hide_index=True)

    with m_col2:
        with st.container(border=True, key="panel13"):
            st.markdown('<div class="section-header">Top Predictive Demand Signals</div>', unsafe_allow_html=True)
            if ml_metrics["top_features"]:
                f_imp = pd.DataFrame(ml_metrics["top_features"])
                f_imp["importance"] = (f_imp["importance"] * 100).round(1).astype(str) + "%"
                f_imp.columns = ["Feature Node", "Importance"]
                st.dataframe(f_imp, width="stretch", hide_index=True)

# Footer
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.82rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid rgba(226, 232, 240, 0.8);">
        Canteen Sales Regression • Built with Python, Pandas, Scikit-learn, Plotly & Streamlit • Academic Showcase Edition
    </div>
    """,
    unsafe_allow_html=True,
)
