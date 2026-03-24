import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from utils.helpers import apply_theme, section_header, metric_card, show_code, log_to_report, PALETTE, check_data
from utils.data_loader import get_rfm

st.set_page_config(page_title="Outlier Detection | DataMineHub",  layout="wide")

section_header("Outlier Detection", "Statistical & proximity-based approaches")

# --- Data ---
ok, raw_df = check_data()
if not ok:
    st.stop()

rfm = get_rfm()

with st.sidebar:
    st.markdown("### Outlier Detection Config")
    data_source = st.radio("Data source", ["Uploaded Dataset", "RFM Features"],
                           index=1 if rfm is not None else 0, key="out_src")

if data_source == "RFM Features" and rfm is not None:
    df = rfm.drop(columns=["CustomerID", "HighValue"], errors="ignore")
else:
    df = raw_df.copy()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

with st.sidebar:
    feature_cols = st.multiselect("Features", numeric_cols,
                                  default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols,
                                  key="out_feats")

if not feature_cols or len(feature_cols) < 1:
    st.info(" Select at least 1 numeric feature from the sidebar.")
    st.stop()

X_raw = df[feature_cols].dropna()
if len(X_raw) < 10:
    st.warning("Not enough data points.")
    st.stop()

# Limit for performance
MAX_SAMPLES = 5000
if len(X_raw) > MAX_SAMPLES:
    X_raw = X_raw.sample(MAX_SAMPLES, random_state=42)
    st.info(f"Sampled {MAX_SAMPLES} points for performance.")

X_scaled = StandardScaler().fit_transform(X_raw)

# --- Method Selection ---
st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#6C63FF,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
section_header("Select Detection Methods", "Check any combination")

col1, col2, col3, col4 = st.columns(4)

methods = {}

with col1:
    if st.checkbox(" Z-Score", value=True, key="out_zscore"):
        z_thresh = st.slider("Z threshold", 1.5, 5.0, 3.0, 0.1, key="out_z_t")
        methods["Z-Score"] = {"threshold": z_thresh}

with col2:
    if st.checkbox(" IQR", value=True, key="out_iqr"):
        iqr_mult = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1, key="out_iqr_m")
        methods["IQR"] = {"multiplier": iqr_mult}

with col3:
    if st.checkbox(" LOF", value=False, key="out_lof"):
        lof_k = st.slider("N neighbors (LOF)", 5, 50, 20, key="out_lof_k")
        methods["LOF"] = {"n_neighbors": lof_k}

with col4:
    if st.checkbox(" Isolation Forest", value=False, key="out_iso"):
        iso_cont = st.slider("Contamination", 0.01, 0.3, 0.05, 0.01, key="out_iso_c")
        methods["Isolation Forest"] = {"contamination": iso_cont}

if not methods:
    st.info(" Select at least one detection method.")
    st.stop()

# --- Run Detection ---
st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#FF6584,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
section_header("Detection Results", "Outliers highlighted in each method")

outlier_masks = {}

for name, params in methods.items():
    if name == "Z-Score":
        z_scores = np.abs((X_raw - X_raw.mean()) / X_raw.std())
        mask = (z_scores > params["threshold"]).any(axis=1)
        outlier_masks["Z-Score"] = mask.values

    elif name == "IQR":
        q1 = X_raw.quantile(0.25)
        q3 = X_raw.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - params["multiplier"] * iqr
        upper = q3 + params["multiplier"] * iqr
        mask = ((X_raw < lower) | (X_raw > upper)).any(axis=1)
        outlier_masks["IQR"] = mask.values

    elif name == "LOF":
        lof = LocalOutlierFactor(n_neighbors=params["n_neighbors"])
        preds = lof.fit_predict(X_scaled)
        outlier_masks["LOF"] = (preds == -1)

    elif name == "Isolation Forest":
        iso = IsolationForest(contamination=params["contamination"], random_state=42)
        preds = iso.fit_predict(X_scaled)
        outlier_masks["Isolation Forest"] = (preds == -1)

# --- Visualize ---
method_tabs = st.tabs(list(outlier_masks.keys()) + [" Agreement"])

# Per-method visualization
for tab, (name, mask) in zip(method_tabs[:-1], outlier_masks.items()):
    with tab:
        n_outliers = mask.sum()
        pct = n_outliers / len(mask) * 100

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Total Points", f"{len(mask):,}", color="primary")
        with c2:
            metric_card("Outliers", f"{n_outliers:,}", color="danger")
        with c3:
            metric_card("Outlier %", f"{pct:.1f}%", color="warning")

        labels = np.where(mask, "Outlier", "Normal")

        # Scatter - use first 2 features or PCA
        if len(feature_cols) >= 2:
            x_col, y_col = feature_cols[0], feature_cols[1]
            fig = px.scatter(X_raw, x=x_col, y=y_col, color=labels,
                             color_discrete_map={"Normal": PALETTE[2], "Outlier": PALETTE[5]},
                             title=f"{name} - Outlier Detection",
                             opacity=0.7)
        else:
            fig = px.scatter(x=range(len(X_raw)), y=X_raw.iloc[:, 0].values,
                             color=labels,
                             color_discrete_map={"Normal": PALETTE[2], "Outlier": PALETTE[5]},
                             title=f"{name} - Outlier Detection",
                             labels={"x": "Index", "y": feature_cols[0]})
        fig = apply_theme(fig)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Box plots
        if len(feature_cols) <= 6:
            plot_data = X_raw.copy()
            plot_data["Outlier"] = labels
            for col in feature_cols:
                fig = px.box(plot_data, y=col, color="Outlier",
                             color_discrete_map={"Normal": PALETTE[2], "Outlier": PALETTE[5]},
                             title=f"{col} - {name}")
                fig = apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        # Code
        if name == "Z-Score":
            show_code(f"""# Z-Score Outlier Detection
z_scores = np.abs((df[{feature_cols}] - df[{feature_cols}].mean()) / df[{feature_cols}].std())
outliers = (z_scores > {methods[name]['threshold']}).any(axis=1)
print(f"Outliers: {{outliers.sum()}} ({{outliers.mean()*100:.1f}}%)")
""")
        elif name == "IQR":
            show_code(f"""# IQR Outlier Detection
Q1 = df[{feature_cols}].quantile(0.25)
Q3 = df[{feature_cols}].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[{feature_cols}] < Q1 - {methods[name]['multiplier']}*IQR) |
            (df[{feature_cols}] > Q3 + {methods[name]['multiplier']}*IQR)).any(axis=1)
print(f"Outliers: {{outliers.sum()}} ({{outliers.mean()*100:.1f}}%)")
""")
        elif name == "LOF":
            show_code(f"""from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors={methods[name]['n_neighbors']})
predictions = lof.fit_predict(X_scaled)
outliers = (predictions == -1)
print(f"Outliers: {{outliers.sum()}} ({{outliers.mean()*100:.1f}}%)")
""")
        elif name == "Isolation Forest":
            show_code(f"""from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination={methods[name]['contamination']}, random_state=42)
predictions = iso.fit_predict(X_scaled)
outliers = (predictions == -1)
print(f"Outliers: {{outliers.sum()}} ({{outliers.mean()*100:.1f}}%)")
""")

# --- Agreement Analysis ---
with method_tabs[-1]:
    section_header("Method Agreement", "Which points do multiple methods flag?")

    if len(outlier_masks) >= 2:
        agreement = pd.DataFrame(outlier_masks).astype(int)
        agreement["Total Methods"] = agreement.sum(axis=1)

        # Agreement distribution
        agree_counts = agreement["Total Methods"].value_counts().sort_index()
        fig = px.bar(x=agree_counts.index, y=agree_counts.values,
                     title="Outlier Agreement Distribution",
                     labels={"x": "# Methods Flagging as Outlier", "y": "# Data Points"},
                     color=agree_counts.index, color_continuous_scale="Reds")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Consensus outliers
        min_agree = st.slider("Minimum method agreement", 1, len(outlier_masks), len(outlier_masks), key="out_agree")
        consensus = agreement["Total Methods"] >= min_agree
        st.markdown(f"**{consensus.sum()}** points flagged by **≥{min_agree}** method(s) ({consensus.sum()/len(consensus)*100:.1f}%)")

        # Impact analysis
        st.markdown("#### Impact of Removing Outliers")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**With outliers**")
            st.dataframe(X_raw.describe().round(3), use_container_width=True)
        with c2:
            st.markdown("**Without consensus outliers**")
            st.dataframe(X_raw[~consensus].describe().round(3), use_container_width=True)

        # Download
        clean_data = X_raw[~consensus]
        csv = clean_data.to_csv(index=False)
        st.download_button(" Download Cleaned Data (outliers removed)", csv,
                           "cleaned_no_outliers.csv", "text/csv", use_container_width=True)

        log_to_report("Outlier Detection",
                      f"<p>Methods: {list(outlier_masks.keys())}</p>" +
                      f"<p>Consensus outliers (≥{min_agree} methods): {consensus.sum()}</p>" +
                      agreement["Total Methods"].value_counts().sort_index().to_frame().to_html())
    else:
        st.info("Select at least 2 methods to see agreement analysis.")
