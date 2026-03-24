import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_default_dataset, preprocess_retail, derive_rfm
from utils.helpers import (
    COLORS, section_header, metric_card, generate_report_html, log_to_report,
)

# --- Page Config ---
st.set_page_config(
    page_title="DataMineHub",
    
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12141D 0%, #0E1117 100%);
    border-right: 1px solid #2A2D3A;
}

/* Gradient header */
.hero-header {
    background: linear-gradient(135deg, #6C63FF 0%, #FF6584 50%, #2ED47A 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0;
}

.hero-sub {
    color: #8B8D97;
    font-size: 1.15rem;
    margin-top: 0.3rem;
    margin-bottom: 2rem;
}

/* Cards */
.feature-card {
    background: linear-gradient(135deg, #1A1D2908, #1A1D2920);
    border: 1px solid #2A2D3A;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.feature-card:hover {
    border-color: #6C63FF;
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
}
.feature-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.feature-title {
    color: #FAFAFA;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.feature-desc {
    color: #8B8D97;
    font-size: 0.88rem;
    line-height: 1.5;
}

/* Status badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-loaded { background: #2ED47A22; color: #2ED47A; border: 1px solid #2ED47A44; }
.badge-empty { background: #FFB94622; color: #FFB946; border: 1px solid #FFB94644; }

/* Hide default Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Divider */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #6C63FF, transparent);
    margin: 2rem 0;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <div style="font-size:2.5rem;"></div>
        <div style="color:#6C63FF;font-weight:700;font-size:1.3rem;letter-spacing:-0.01em;">DataMineHub</div>
        <div style="color:#8B8D97;font-size:0.8rem;">Interactive Data Mining Workbench</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Dataset status
    if "dataset" in st.session_state and st.session_state["dataset"] is not None:
        n_rows = len(st.session_state["dataset"])
        n_cols = len(st.session_state["dataset"].columns)
        st.markdown(f'<span class="badge badge-loaded"> Dataset Loaded</span> &nbsp; {n_rows:,} x {n_cols}', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-empty"> No Dataset</span>', unsafe_allow_html=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Report download
    if st.button(" Generate Report", use_container_width=True):
        report_html = generate_report_html()
        st.download_button(
            label=" Download Report (HTML)",
            data=report_html,
            file_name="dataminehub_report.html",
            mime="text/html",
            use_container_width=True,
        )

# --- Hero Section ---
st.markdown('<div class="hero-header">DataMineHub</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">An interactive data mining workbench - upload data, tweak parameters, explore algorithms, and export results. Built for learning and demonstration.</div>', unsafe_allow_html=True)

# --- Dataset Upload ---
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
section_header("Load Your Dataset", "Upload a CSV or use the built-in Online Retail dataset")

col_upload, col_default = st.columns(2)

with col_upload:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon"></div>
        <div class="feature-title">Upload Custom CSV</div>
        <div class="feature-desc">Bring your own dataset - any CSV file works</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose CSV", type=["csv"], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["dataset"] = df
            st.session_state["dataset_name"] = uploaded_file.name
            st.session_state["is_retail"] = False
            # Try to derive RFM if possible
            cleaned = preprocess_retail(df)
            st.session_state["cleaned_data"] = cleaned
            rfm = derive_rfm(cleaned)
            st.session_state["rfm_data"] = rfm
            st.success(f" Loaded **{uploaded_file.name}** - {df.shape[0]:,} rows x {df.shape[1]} columns")
            log_to_report("Dataset", f"<p>Uploaded custom dataset: <strong>{uploaded_file.name}</strong> ({df.shape[0]:,} rows x {df.shape[1]} cols)</p>")
        except Exception as e:
            st.error(f" Error reading file: {e}")

with col_default:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon"></div>
        <div class="feature-title">Online Retail Dataset</div>
        <div class="feature-desc">541K+ transactions from a UK online retailer - perfect for all DM techniques</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(" Load Default Dataset", use_container_width=True):
        with st.spinner("Loading & preprocessing..."):
            df = load_default_dataset()
            if df is not None:
                st.session_state["dataset"] = df
                st.session_state["dataset_name"] = "Online Retail"
                st.session_state["is_retail"] = True
                cleaned = preprocess_retail(df)
                st.session_state["cleaned_data"] = cleaned
                rfm = derive_rfm(cleaned)
                st.session_state["rfm_data"] = rfm
                st.success(f" Loaded **Online Retail** - {df.shape[0]:,} rows x {df.shape[1]} columns")
                log_to_report("Dataset", f"<p>Loaded default Online Retail dataset ({df.shape[0]:,} rows x {df.shape[1]} cols)</p>")
            else:
                st.error(" Default dataset not found in `data/` folder. Please upload a CSV instead.")

# --- Dataset Preview ---
if "dataset" in st.session_state and st.session_state["dataset"] is not None:
    df = st.session_state["dataset"]

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    section_header("Dataset Preview", f"Exploring: **{st.session_state.get('dataset_name', 'Unknown')}**")

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Rows", f"{df.shape[0]:,}", color="primary")
    with c2:
        metric_card("Columns", f"{df.shape[1]}", color="secondary")
    with c3:
        metric_card("Missing Values", f"{df.isnull().sum().sum():,}", color="warning")
    with c4:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metric_card("Numeric Columns", f"{len(numeric_cols)}", color="success")

    # Show data
    tab_head, tab_tail, tab_dtypes, tab_stats = st.tabs([" First Rows", " Last Rows", " Data Types", " Quick Stats"])

    with tab_head:
        st.dataframe(df.head(20), use_container_width=True, height=400)

    with tab_tail:
        st.dataframe(df.tail(20), use_container_width=True, height=400)

    with tab_dtypes:
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Null": df.isnull().sum().values,
            "Null %": (df.isnull().sum().values / len(df) * 100).round(2),
            "Unique": df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, height=400)

    with tab_stats:
        st.dataframe(df.describe(include="all").T, use_container_width=True, height=400)

    # RFM info
    if st.session_state.get("rfm_data") is not None:
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
        section_header("Derived: Customer RFM Features", "Auto-generated from transaction data for classification, clustering & outlier detection")
        rfm = st.session_state["rfm_data"]
        st.dataframe(rfm.head(10), use_container_width=True)
        st.caption(f"{rfm.shape[0]:,} customers | Features: Recency, Frequency, Monetary, AvgOrderValue, HighValue (target)")

# --- Feature Cards ---
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
section_header("Explore the Workbench", "Each page covers a major data mining technique - click in the sidebar to navigate")

features = [
    ("", "Data Overview", "Attribute types, statistical descriptions, similarity & dissimilarity measures", "primary"),
    ("", "Preprocessing", "Cleaning, normalization, discretization, PCA with before/after views", "secondary"),
    ("", "Association Rules", "Apriori algorithm, frequent itemsets, support, confidence, lift", "success"),
    ("", "Classification", "Decision Tree, Naive Bayes, SVM, KNN, Random Forest, ensembles & model comparison", "warning"),
    ("", "Clustering", "K-Means, Hierarchical, DBSCAN with live parameter tuning", "info"),
    ("", "Outlier Detection", "Z-Score, IQR, LOF, Isolation Forest with visual comparison", "danger"),
    ("", "Playground", "Generate synthetic datasets (moons, blobs, circles) and stress-test algorithms", "primary"),
]

cols = st.columns(3)
for i, (icon, title, desc, color) in enumerate(features):
    with cols[i % 3]:
        c = COLORS.get(color, "#6C63FF")
        st.markdown(f"""
        <div class="feature-card" style="border-left: 3px solid {c};">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1rem 0;color:#8B8D97;font-size:0.8rem;">
    DataMineHub | Interactive Data Mining Workbench | Built with Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)
