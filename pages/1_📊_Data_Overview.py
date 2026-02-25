import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew, kurtosis
from utils.helpers import apply_theme, section_header, metric_card, show_code, log_to_report, PALETTE, check_data

st.set_page_config(page_title="Data Overview | DataMineHub", page_icon="📊", layout="wide")

# ─── Data Check ──────────────────────────────────────────────────────────────

ok, df = check_data()
if not ok:
    st.stop()

section_header("Data Overview & EDA", "Attribute types, statistical descriptions, similarity & dissimilarity", "📊")

# ─── Column Selector ─────────────────────────────────────────────────────────

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
all_cols = df.columns.tolist()

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    selected_cols = st.multiselect("Select columns to analyze", all_cols, default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols)

if not selected_cols:
    st.info("👈 Select at least one column from the sidebar.")
    st.stop()

sub_df = df[selected_cols]

# ─── Attribute Types ─────────────────────────────────────────────────────────

section_header("Attribute Classification", "Automatic detection of attribute types", "🏷️")

attr_rows = []
for col in selected_cols:
    dtype = df[col].dtype
    nunique = df[col].nunique()
    if pd.api.types.is_numeric_dtype(dtype):
        if nunique <= 10:
            atype = "Ordinal (Numeric, ≤10 unique)"
        elif pd.api.types.is_integer_dtype(dtype) and nunique <= 50:
            atype = "Discrete / Ratio"
        else:
            atype = "Continuous / Ratio"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        atype = "Interval (Datetime)"
    else:
        if nunique <= 2:
            atype = "Binary / Nominal"
        elif nunique <= 20:
            atype = "Nominal (Low Cardinality)"
        else:
            atype = "Nominal (High Cardinality)"
    attr_rows.append({"Column": col, "Data Type": str(dtype), "Unique Values": nunique, "Attribute Type": atype})

attr_df = pd.DataFrame(attr_rows)
st.dataframe(attr_df, use_container_width=True, hide_index=True)

show_code(f"""import pandas as pd

df = pd.read_csv('your_data.csv')

# Check data types
print(df[{selected_cols}].dtypes)
print(df[{selected_cols}].nunique())
""")

log_to_report("Data Overview — Attribute Types", attr_df.to_html(index=False))

# ─── Statistical Descriptions ────────────────────────────────────────────────

st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#6C63FF,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
section_header("Statistical Descriptions", "Central tendency, spread, and shape measures", "📈")

num_selected = [c for c in selected_cols if c in numeric_cols]
if num_selected:
    stats_rows = []
    for col in num_selected:
        s = df[col].dropna()
        stats_rows.append({
            "Column": col,
            "Count": int(s.count()),
            "Mean": round(s.mean(), 4),
            "Median": round(s.median(), 4),
            "Mode": round(s.mode().iloc[0], 4) if len(s.mode()) > 0 else "N/A",
            "Std Dev": round(s.std(), 4),
            "Variance": round(s.var(), 4),
            "Min": round(s.min(), 4),
            "Q1 (25%)": round(s.quantile(0.25), 4),
            "Q3 (75%)": round(s.quantile(0.75), 4),
            "Max": round(s.max(), 4),
            "IQR": round(s.quantile(0.75) - s.quantile(0.25), 4),
            "Skewness": round(skew(s), 4),
            "Kurtosis": round(kurtosis(s), 4),
        })
    stats_df = pd.DataFrame(stats_rows)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    show_code(f"""from scipy.stats import skew, kurtosis

cols = {num_selected}
for col in cols:
    s = df[col].dropna()
    print(f"{{col}}:")
    print(f"  Mean: {{s.mean():.4f}}, Median: {{s.median():.4f}}")
    print(f"  Std: {{s.std():.4f}}, Variance: {{s.var():.4f}}")
    print(f"  Q1: {{s.quantile(0.25):.4f}}, Q3: {{s.quantile(0.75):.4f}}")
    print(f"  Skewness: {{skew(s):.4f}}, Kurtosis: {{kurtosis(s):.4f}}")
""")

    log_to_report("Data Overview — Statistics", stats_df.to_html(index=False))

    # ─── Distribution Charts ─────────────────────────────────────────────────

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#FF6584,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
    section_header("Distributions", "Interactive visualizations — pick chart type", "📊")

    chart_type = st.selectbox("Chart Type", ["Histogram", "Box Plot", "Violin Plot"], key="chart_type_eda")
    vis_cols = st.multiselect("Columns to plot", num_selected, default=num_selected[:3], key="vis_cols_eda")

    if vis_cols:
        if chart_type == "Histogram":
            n_bins = st.slider("Number of bins", 10, 100, 30, key="nbins_eda")
            for col in vis_cols:
                fig = px.histogram(df, x=col, nbins=n_bins, title=f"Distribution of {col}",
                                   color_discrete_sequence=[PALETTE[0]], opacity=0.85)
                fig = apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Box Plot":
            fig = px.box(df[vis_cols].melt(), x="variable", y="value", color="variable",
                         title="Box Plots", color_discrete_sequence=PALETTE)
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.violin(df[vis_cols].melt(), x="variable", y="value", color="variable",
                            box=True, title="Violin Plots", color_discrete_sequence=PALETTE)
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    # ─── Correlation Heatmap ─────────────────────────────────────────────────

    if len(num_selected) >= 2:
        st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#2ED47A,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
        section_header("Correlation Matrix", "Relationships between numeric features", "🔥")

        corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], key="corr_method")
        corr = df[num_selected].corr(method=corr_method)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title=f"{corr_method.title()} Correlation Matrix", aspect="auto")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        show_code(f"""import plotly.express as px

corr = df[{num_selected}].corr(method='{corr_method}')
fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r')
fig.show()
""")

    # ─── Similarity / Dissimilarity ──────────────────────────────────────────

    if len(num_selected) >= 2:
        st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#FFB946,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
        section_header("Similarity & Dissimilarity", "Compute pairwise distance matrices", "📏")

        metric = st.selectbox("Distance / Similarity Metric",
                              ["euclidean", "cosine", "cityblock", "chebyshev", "minkowski"],
                              key="dist_metric")

        n_samples = st.slider("Number of samples (for performance)", 10, min(500, len(df)), min(50, len(df)), key="n_samples_sim")

        sampled = df[num_selected].dropna().sample(n=min(n_samples, len(df[num_selected].dropna())), random_state=42)

        if len(sampled) > 1:
            dist_matrix = squareform(pdist(sampled.values, metric=metric))
            dist_df = pd.DataFrame(dist_matrix, index=sampled.index, columns=sampled.index)

            fig = px.imshow(dist_matrix, color_continuous_scale="Viridis",
                            title=f"Pairwise {metric.title()} Distance ({n_samples} samples)", aspect="auto")
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            show_code(f"""from scipy.spatial.distance import pdist, squareform

data = df[{num_selected}].dropna().sample(n={n_samples}, random_state=42)
dist = squareform(pdist(data.values, metric='{metric}'))

import plotly.express as px
fig = px.imshow(dist, color_continuous_scale='Viridis',
                title='Pairwise {metric.title()} Distance')
fig.show()
""")

            log_to_report("Data Overview — Similarity", f"<p>Computed {metric} distance matrix on {n_samples} samples across {len(num_selected)} features.</p>")
else:
    st.info("Select at least one numeric column to see statistical descriptions and visualizations.")
