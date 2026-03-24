import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# --- Consistent Plotly Theme ---
COLORS = {
    "primary": "#6C63FF",
    "secondary": "#FF6584",
    "success": "#2ED47A",
    "warning": "#FFB946",
    "info": "#56CCF2",
    "danger": "#EB5757",
    "bg": "#0E1117",
    "card_bg": "#1A1D29",
    "text": "#FAFAFA",
    "muted": "#8B8D97",
}

PALETTE = [
    "#6C63FF", "#FF6584", "#2ED47A", "#FFB946", "#56CCF2",
    "#EB5757", "#BB6BD9", "#F2994A", "#27AE60", "#2F80ED",
    "#E84393", "#00CEC9", "#FDCB6E", "#A29BFE", "#FD79A8",
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA", family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
    colorway=PALETTE,
)


def apply_theme(fig):
    """Apply consistent dark theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#2A2D3A", zerolinecolor="#2A2D3A")
    fig.update_yaxes(gridcolor="#2A2D3A", zerolinecolor="#2A2D3A")
    return fig


# --- Metric Cards ---
def metric_card(label, value, delta=None, color="primary"):
    """Render a styled metric card."""
    c = COLORS.get(color, color)
    delta_html = ""
    if delta is not None:
        arrow = "" if delta > 0 else "" if delta < 0 else ""
        delta_color = "#2ED47A" if delta > 0 else "#EB5757" if delta < 0 else "#8B8D97"
        delta_html = f'<span style="color:{delta_color};font-size:0.85rem;">{arrow} {abs(delta):.2f}</span>'

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {c}22, {c}08);
        border-left: 4px solid {c};
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.5rem;
    ">
        <div style="color:#8B8D97;font-size:0.82rem;text-transform:uppercase;letter-spacing:0.05em;">{label}</div>
        <div style="color:#FAFAFA;font-size:1.7rem;font-weight:700;margin-top:0.15rem;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(title, subtitle=None, icon=""):
    """Render a styled section header."""
    sub = f'<p style="color:#8B8D97;margin-top:0.2rem;font-size:0.95rem;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom:1.2rem;">
        <h2 style="color:#FAFAFA;margin-bottom:0;">{icon} {title}</h2>
        {sub}
    </div>
    """, unsafe_allow_html=True)


# --- Code Mirror ---
def show_code(code_string, language="python"):
    """Show a collapsible Python code snippet (Live Code Mirror)."""
    with st.expander(" View Python Code", expanded=False):
        st.code(code_string.strip(), language=language)


# --- Report Helpers ---
def log_to_report(section, content):
    """Append content to the session-state report log."""
    if "report_log" not in st.session_state:
        st.session_state["report_log"] = []
    st.session_state["report_log"].append({"section": section, "content": content})


def generate_report_html():
    """Generate an HTML report from all logged sections."""
    if "report_log" not in st.session_state or not st.session_state["report_log"]:
        return "<h1>DataMineHub Report</h1><p>No analysis has been performed yet. Visit the pages and interact with the tools to generate report content.</p>"

    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DataMineHub - Analysis Report</title>
<style>
body { font-family: 'Segoe UI', sans-serif; background: #0E1117; color: #FAFAFA; max-width: 900px; margin: 0 auto; padding: 2rem; }
h1 { color: #6C63FF; border-bottom: 2px solid #6C63FF; padding-bottom: 0.5rem; }
h2 { color: #FF6584; margin-top: 2rem; }
pre { background: #1A1D29; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.9rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #2A2D3A; padding: 0.6rem; text-align: left; }
th { background: #1A1D29; color: #6C63FF; }
.metric { display: inline-block; background: #1A1D29; border-radius: 8px; padding: 0.8rem 1.2rem; margin: 0.3rem; }
.metric .label { color: #8B8D97; font-size: 0.8rem; }
.metric .value { color: #FAFAFA; font-size: 1.3rem; font-weight: bold; }
</style>
</head>
<body>
<h1> DataMineHub - Analysis Report</h1>
<p style="color:#8B8D97;">Auto-generated report of data mining analysis session.</p>
"""
    for entry in st.session_state["report_log"]:
        html += f"<h2>{entry['section']}</h2>\n"
        html += f"<div>{entry['content']}</div>\n"

    html += "</body></html>"
    return html


# --- Data Checks ---
def check_data(require_numeric=False, min_cols=1):
    """Check that data is loaded and valid. Returns (True, df) or (False, None)."""
    from utils.data_loader import get_data
    df = get_data()
    if df is None:
        st.warning(" No dataset loaded. Go to the **Home** page and upload a CSV or load the default dataset.")
        return False, None
    if require_numeric:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < min_cols:
            st.warning(f" Need at least {min_cols} numeric column(s). Found {len(num_cols)}.")
            return False, None
    return True, df
