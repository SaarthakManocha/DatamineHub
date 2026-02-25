import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from utils.helpers import apply_theme, section_header, metric_card, show_code, log_to_report, PALETTE, check_data

st.set_page_config(page_title="Preprocessing | DataMineHub", page_icon="🧹", layout="wide")

ok, df = check_data()
if not ok:
    st.stop()

section_header("Data Preprocessing", "Cleaning, transformation, reduction & discretization", "🧹")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.warning("No numeric columns found in the dataset.")
    st.stop()

# Initialize processed data
if "preprocessed" not in st.session_state:
    st.session_state["preprocessed"] = None

# ─── Sidebar Controls ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Preprocessing Pipeline")

    selected_cols = st.multiselect("Columns to process", numeric_cols, default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols, key="pp_cols")

    st.markdown("---")
    st.markdown("**Toggle Steps:**")
    do_missing = st.checkbox("Handle Missing Values", value=True, key="pp_missing")
    do_outliers = st.checkbox("Handle Outliers", value=False, key="pp_outliers")
    do_normalize = st.checkbox("Normalize / Scale", value=False, key="pp_normalize")
    do_discretize = st.checkbox("Discretize", value=False, key="pp_discretize")
    do_pca = st.checkbox("PCA Reduction", value=False, key="pp_pca")

if not selected_cols:
    st.info("👈 Select columns from the sidebar to start preprocessing.")
    st.stop()

working = df[selected_cols].copy()
report_steps = []

# ─── 1. Missing Values ──────────────────────────────────────────────────────

st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#6C63FF,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
section_header("1. Missing Values", "Identify and handle missing data", "🕳️")

missing_counts = working.isnull().sum()
missing_pct = (missing_counts / len(working) * 100).round(2)
missing_df = pd.DataFrame({"Column": selected_cols, "Missing": missing_counts.values, "% Missing": missing_pct.values})

c1, c2 = st.columns([1, 1])
with c1:
    st.dataframe(missing_df, use_container_width=True, hide_index=True)
with c2:
    fig = px.bar(missing_df[missing_df["Missing"] > 0], x="Column", y="Missing",
                 color="% Missing", color_continuous_scale="Reds",
                 title="Missing Values per Column")
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

if do_missing and missing_counts.sum() > 0:
    strategy = st.radio("Imputation strategy", ["mean", "median", "most_frequent", "drop rows"], horizontal=True, key="imp_strategy")

    if strategy == "drop rows":
        before_count = len(working)
        working = working.dropna()
        after_count = len(working)
        st.success(f"Dropped {before_count - after_count} rows → {after_count} remaining")
        report_steps.append(f"Dropped {before_count - after_count} rows with missing values.")

        show_code(f"""# Drop rows with missing values
df_clean = df[{selected_cols}].dropna()
print(f"Dropped {{len(df) - len(df_clean)}} rows")
""")
    else:
        imputer = SimpleImputer(strategy=strategy)
        working = pd.DataFrame(imputer.fit_transform(working), columns=selected_cols, index=working.index)
        st.success(f"Imputed missing values using **{strategy}**")
        report_steps.append(f"Imputed missing values using {strategy}.")

        show_code(f"""from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='{strategy}')
df_imputed = pd.DataFrame(
    imputer.fit_transform(df[{selected_cols}]),
    columns={selected_cols}
)
""")
elif missing_counts.sum() == 0:
    st.success("✅ No missing values found!")

# ─── 2. Outlier Handling ─────────────────────────────────────────────────────

if do_outliers:
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#FF6584,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
    section_header("2. Outlier Handling", "Detect and cap outliers", "🔧")

    outlier_method = st.radio("Detection method", ["IQR", "Z-Score"], horizontal=True, key="outlier_method")

    if outlier_method == "IQR":
        iqr_mult = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1, key="iqr_mult")
    else:
        z_thresh = st.slider("Z-Score threshold", 1.0, 5.0, 3.0, 0.1, key="z_thresh_pp")

    # Before
    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown("**Before**")
        fig = px.box(working.melt(), x="variable", y="value", color="variable",
                     color_discrete_sequence=PALETTE, title="Before Outlier Handling")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Apply
    outlier_count = 0
    for col in working.select_dtypes(include=[np.number]).columns:
        if outlier_method == "IQR":
            q1 = working[col].quantile(0.25)
            q3 = working[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_mult * iqr
            upper = q3 + iqr_mult * iqr
        else:
            mean = working[col].mean()
            std = working[col].std()
            lower = mean - z_thresh * std
            upper = mean + z_thresh * std

        outliers = (working[col] < lower) | (working[col] > upper)
        outlier_count += outliers.sum()
        working[col] = working[col].clip(lower, upper)

    with col_a:
        st.markdown("**After**")
        fig = px.box(working.melt(), x="variable", y="value", color="variable",
                     color_discrete_sequence=PALETTE, title="After Outlier Handling")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.success(f"Capped **{outlier_count}** outlier values using {outlier_method}")
    report_steps.append(f"Capped {outlier_count} outliers using {outlier_method}.")

    if outlier_method == "IQR":
        show_code(f"""# IQR-based outlier capping
for col in df.select_dtypes(include='number').columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - {iqr_mult}*iqr, q3 + {iqr_mult}*iqr
    df[col] = df[col].clip(lower, upper)
""")
    else:
        show_code(f"""# Z-Score outlier capping
for col in df.select_dtypes(include='number').columns:
    mean, std = df[col].mean(), df[col].std()
    lower, upper = mean - {z_thresh}*std, mean + {z_thresh}*std
    df[col] = df[col].clip(lower, upper)
""")

# ─── 3. Normalization ────────────────────────────────────────────────────────

if do_normalize:
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#2ED47A,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
    section_header("3. Normalization / Scaling", "Scale features to a common range", "⚖️")

    norm_method = st.radio("Method", ["Min-Max (0-1)", "Z-Score (StandardScaler)", "Robust Scaler"], horizontal=True, key="norm_method")

    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown("**Before**")
        fig = px.histogram(working.melt(), x="value", color="variable", barmode="overlay",
                           opacity=0.6, color_discrete_sequence=PALETTE, title="Before Scaling")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    if norm_method == "Min-Max (0-1)":
        scaler = MinMaxScaler()
        scaler_name = "MinMaxScaler"
    elif norm_method == "Z-Score (StandardScaler)":
        scaler = StandardScaler()
        scaler_name = "StandardScaler"
    else:
        scaler = RobustScaler()
        scaler_name = "RobustScaler"

    num_work_cols = working.select_dtypes(include=[np.number]).columns
    working[num_work_cols] = scaler.fit_transform(working[num_work_cols])

    with col_a:
        st.markdown("**After**")
        fig = px.histogram(working.melt(), x="value", color="variable", barmode="overlay",
                           opacity=0.6, color_discrete_sequence=PALETTE, title=f"After {scaler_name}")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.success(f"Applied **{scaler_name}**")
    report_steps.append(f"Normalized using {scaler_name}.")

    show_code(f"""from sklearn.preprocessing import {scaler_name}

scaler = {scaler_name}()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[{list(num_work_cols)}]),
    columns={list(num_work_cols)}
)
""")

# ─── 4. Discretization ───────────────────────────────────────────────────────

if do_discretize:
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#FFB946,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
    section_header("4. Discretization", "Convert continuous features into bins", "📦")

    disc_method = st.radio("Binning method", ["Equal Width (Uniform)", "Equal Frequency (Quantile)"], horizontal=True, key="disc_method")
    n_bins = st.slider("Number of bins", 2, 20, 5, key="disc_bins")
    disc_col = st.selectbox("Column to discretize", working.select_dtypes(include=[np.number]).columns.tolist(), key="disc_col")

    strategy = "uniform" if "Width" in disc_method else "quantile"

    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown("**Continuous**")
        fig = px.histogram(working, x=disc_col, nbins=40, title=f"{disc_col} — Continuous",
                           color_discrete_sequence=[PALETTE[0]])
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    try:
        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
        col_data = working[[disc_col]].dropna()
        if len(col_data) > 0:
            binned = disc.fit_transform(col_data)
            working.loc[col_data.index, disc_col + "_binned"] = binned.flatten()

            with col_a:
                st.markdown(f"**Discretized ({n_bins} bins)**")
                fig = px.histogram(working.dropna(subset=[disc_col + "_binned"]),
                                   x=disc_col + "_binned", nbins=n_bins,
                                   title=f"{disc_col} — {disc_method} ({n_bins} bins)",
                                   color_discrete_sequence=[PALETTE[1]])
                fig = apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            report_steps.append(f"Discretized {disc_col} into {n_bins} bins using {strategy}.")

            show_code(f"""from sklearn.preprocessing import KBinsDiscretizer

disc = KBinsDiscretizer(n_bins={n_bins}, encode='ordinal', strategy='{strategy}')
df['{disc_col}_binned'] = disc.fit_transform(df[['{disc_col}']])
""")
    except Exception as e:
        st.error(f"Discretization error: {e}")

# ─── 5. PCA ──────────────────────────────────────────────────────────────────

if do_pca and len(working.select_dtypes(include=[np.number]).columns) >= 2:
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#56CCF2,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
    section_header("5. Dimensionality Reduction (PCA)", "Reduce features while preserving variance", "🔽")

    num_work_cols = working.select_dtypes(include=[np.number]).columns.tolist()
    max_comp = min(len(num_work_cols), 10, len(working.dropna()) - 1)
    if max_comp < 2:
        st.info("Need at least 2 numeric columns and enough data for PCA.")
    else:
        n_components = st.slider("Number of components", 2, max_comp, min(2, max_comp), key="pca_comp")

        clean_for_pca = working[num_work_cols].dropna()
        if len(clean_for_pca) > 1:
            # Standardize before PCA
            from sklearn.preprocessing import StandardScaler as SS
            pca_data = SS().fit_transform(clean_for_pca)

            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(pca_data)

            # Explained variance
            var_ratio = pca.explained_variance_ratio_
            cum_var = np.cumsum(var_ratio)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(x=[f"PC{i+1}" for i in range(n_components)], y=var_ratio,
                             title="Explained Variance per Component",
                             labels={"x": "Component", "y": "Explained Variance Ratio"},
                             color_discrete_sequence=[PALETTE[4]])
                fig = apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = px.line(x=[f"PC{i+1}" for i in range(n_components)], y=cum_var,
                              title="Cumulative Explained Variance",
                              labels={"x": "Component", "y": "Cumulative Variance"},
                              markers=True)
                fig.update_traces(line_color=PALETTE[0])
                fig = apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            st.info(f"**{n_components} components** explain **{cum_var[-1]*100:.1f}%** of total variance (reduced from {len(num_work_cols)} features)")

            # 2D scatter of first 2 PCs
            if n_components >= 2:
                pca_df = pd.DataFrame(transformed[:, :2], columns=["PC1", "PC2"])
                fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA — First 2 Components",
                                 color_discrete_sequence=[PALETTE[0]], opacity=0.6)
                fig = apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            report_steps.append(f"PCA: {n_components} components, {cum_var[-1]*100:.1f}% variance explained.")

            show_code(f"""from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = StandardScaler().fit_transform(df[{num_work_cols}].dropna())
pca = PCA(n_components={n_components})
transformed = pca.fit_transform(data)

print(f"Explained variance: {{pca.explained_variance_ratio_}}")
print(f"Total: {{sum(pca.explained_variance_ratio_)*100:.1f}}%")
""")

# ─── Download Preprocessed Data ──────────────────────────────────────────────

st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#6C63FF,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
section_header("📥 Download Preprocessed Data", "Export the transformed dataset", "")

st.session_state["preprocessed"] = working
st.dataframe(working.head(20), use_container_width=True)

csv = working.to_csv(index=False)
st.download_button("⬇️ Download as CSV", csv, "preprocessed_data.csv", "text/csv", use_container_width=True)

if report_steps:
    log_to_report("Preprocessing", "<ul>" + "".join(f"<li>{s}</li>" for s in report_steps) + "</ul>")
