import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from utils.helpers import apply_theme, section_header, metric_card, show_code, log_to_report, PALETTE, check_data
from utils.data_loader import get_rfm

st.set_page_config(page_title="Clustering | DataMineHub", page_icon="📦", layout="wide")

section_header("Cluster Analysis", "K-Means, Hierarchical, DBSCAN with live parameter tuning", "📦")

# ─── Data ─────────────────────────────────────────────────────────────────────

ok, raw_df = check_data()
if not ok:
    st.stop()

rfm = get_rfm()

with st.sidebar:
    st.markdown("### ⚙️ Clustering Config")
    data_source = st.radio("Data source", ["Uploaded Dataset", "RFM Features"],
                           index=1 if rfm is not None else 0, key="clust_src")

if data_source == "RFM Features" and rfm is not None:
    df = rfm.drop(columns=["CustomerID", "HighValue"], errors="ignore")
else:
    df = raw_df.copy()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

with st.sidebar:
    feature_cols = st.multiselect("Features for clustering", numeric_cols,
                                  default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols,
                                  key="clust_feats")
    viz_mode = st.radio("Visualization", ["2D (PCA)", "3D (PCA)"], key="clust_viz")

if not feature_cols or len(feature_cols) < 2:
    st.info("👈 Select at least 2 numeric features from the sidebar.")
    st.stop()

# Prepare data
X_raw = df[feature_cols].dropna()
if len(X_raw) < 10:
    st.warning("Not enough data points for meaningful clustering.")
    st.stop()

# Limit samples for performance
MAX_SAMPLES = 5000
if len(X_raw) > MAX_SAMPLES:
    X_raw = X_raw.sample(MAX_SAMPLES, random_state=42)
    st.info(f"Sampled {MAX_SAMPLES} points for performance.")

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# PCA for visualization
n_comp = 3 if viz_mode == "3D (PCA)" else 2
pca = PCA(n_components=min(n_comp, X.shape[1]))
X_pca = pca.fit_transform(X)

# ─── Algorithm Tabs ──────────────────────────────────────────────────────────

tab_km, tab_hier, tab_db, tab_compare = st.tabs(["📍 K-Means", "🏔️ Hierarchical", "🔷 DBSCAN", "🏆 Compare"])

results = {}

# ═══════════════════════════════════════════════════════════════════════════
# K-MEANS
# ═══════════════════════════════════════════════════════════════════════════

with tab_km:
    section_header("K-Means Clustering", "Partitioning method — adjust K and see clusters reform", "📍")

    c_left, c_right = st.columns([1, 2])

    with c_left:
        k = st.slider("Number of clusters (K)", 2, 15, 3, key="km_k")
        n_init = st.slider("Number of initializations", 1, 20, 10, key="km_ninit")
        show_elbow = st.checkbox("Show Elbow Method", value=True, key="km_elbow")

    # Train
    km = KMeans(n_clusters=k, n_init=n_init, random_state=42)
    labels_km = km.fit_predict(X)
    sil_km = silhouette_score(X, labels_km) if len(set(labels_km)) > 1 else 0
    results["K-Means"] = {"labels": labels_km, "silhouette": sil_km, "k": k}

    with c_left:
        metric_card("Silhouette Score", f"{sil_km:.4f}", color="primary")
        metric_card("Inertia", f"{km.inertia_:.1f}", color="secondary")

    with c_right:
        if viz_mode == "3D (PCA)" and X_pca.shape[1] >= 3:
            fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                                color=labels_km.astype(str), title=f"K-Means (K={k})",
                                labels={"x": "PC1", "y": "PC2", "z": "PC3"},
                                color_discrete_sequence=PALETTE)
        else:
            fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=labels_km.astype(str),
                             title=f"K-Means (K={k})", labels={"x": "PC1", "y": "PC2"},
                             color_discrete_sequence=PALETTE)
        fig = apply_theme(fig)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Elbow method
    if show_elbow:
        st.markdown("#### 📐 Elbow Method")
        inertias = []
        sils = []
        k_range = range(2, min(12, len(X)))
        for ki in k_range:
            kmi = KMeans(n_clusters=ki, n_init=5, random_state=42)
            kmi.fit(X)
            inertias.append(kmi.inertia_)
            sils.append(silhouette_score(X, kmi.labels_))

        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(x=list(k_range), y=inertias, markers=True,
                          title="Elbow Method (Inertia)", labels={"x": "K", "y": "Inertia"})
            fig.update_traces(line_color=PALETTE[0])
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(x=list(k_range), y=sils, markers=True,
                          title="Silhouette Scores", labels={"x": "K", "y": "Silhouette"})
            fig.update_traces(line_color=PALETTE[1])
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Cluster profiling
    st.markdown("#### 📋 Cluster Profiles")
    profile = X_raw.copy()
    profile["Cluster"] = labels_km
    st.dataframe(profile.groupby("Cluster").agg(["mean", "std", "count"]).round(2),
                 use_container_width=True)

    show_code(f"""from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

X = StandardScaler().fit_transform(df[{feature_cols}].dropna())
km = KMeans(n_clusters={k}, n_init={n_init}, random_state=42)
labels = km.fit_predict(X)
print(f"Silhouette: {{silhouette_score(X, labels):.4f}}")
print(f"Inertia: {{km.inertia_:.1f}}")
""")

# ═══════════════════════════════════════════════════════════════════════════
# HIERARCHICAL
# ═══════════════════════════════════════════════════════════════════════════

with tab_hier:
    section_header("Hierarchical Clustering", "Agglomerative — with dendrogram", "🏔️")

    c_left, c_right = st.columns([1, 2])

    with c_left:
        h_k = st.slider("Number of clusters", 2, 15, 3, key="hier_k")
        h_linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"], key="hier_link")
        h_metric_opt = "euclidean" if h_linkage == "ward" else st.selectbox("Distance metric", 
            ["euclidean", "manhattan", "cosine"], key="hier_met")

    # Train
    agg = AgglomerativeClustering(n_clusters=h_k, linkage=h_linkage, metric=h_metric_opt if h_linkage != "ward" else "euclidean")
    labels_hier = agg.fit_predict(X)
    sil_hier = silhouette_score(X, labels_hier) if len(set(labels_hier)) > 1 else 0
    results["Hierarchical"] = {"labels": labels_hier, "silhouette": sil_hier, "k": h_k}

    with c_left:
        metric_card("Silhouette Score", f"{sil_hier:.4f}", color="success")

    with c_right:
        if viz_mode == "3D (PCA)" and X_pca.shape[1] >= 3:
            fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                                color=labels_hier.astype(str), title=f"Hierarchical ({h_linkage}, k={h_k})",
                                color_discrete_sequence=PALETTE)
        else:
            fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=labels_hier.astype(str),
                             title=f"Hierarchical ({h_linkage}, k={h_k})",
                             labels={"x": "PC1", "y": "PC2"},
                             color_discrete_sequence=PALETTE)
        fig = apply_theme(fig)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Dendrogram
    st.markdown("#### 🌳 Dendrogram")
    max_dendro_samples = min(100, len(X))
    X_dendro = X[:max_dendro_samples]
    Z = linkage(X_dendro, method=h_linkage)
    fig_dendro, ax = plt.subplots(figsize=(12, 4))
    ax.set_facecolor("#0E1117")
    fig_dendro.patch.set_facecolor("#0E1117")
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, leaf_rotation=90,
               leaf_font_size=8, color_threshold=0)
    ax.tick_params(colors="#FAFAFA")
    ax.spines["bottom"].set_color("#2A2D3A")
    ax.spines["left"].set_color("#2A2D3A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Dendrogram", color="#FAFAFA", fontsize=14)
    st.pyplot(fig_dendro)
    plt.close()

    show_code(f"""from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

X = StandardScaler().fit_transform(df[{feature_cols}].dropna())
agg = AgglomerativeClustering(n_clusters={h_k}, linkage='{h_linkage}')
labels = agg.fit_predict(X)
print(f"Silhouette: {{silhouette_score(X, labels):.4f}}")

# Dendrogram
Z = linkage(X[:100], method='{h_linkage}')
dendrogram(Z, truncate_mode='lastp', p=30)
""")

# ═══════════════════════════════════════════════════════════════════════════
# DBSCAN
# ═══════════════════════════════════════════════════════════════════════════

with tab_db:
    section_header("DBSCAN", "Density-based — detects arbitrary shapes + noise", "🔷")

    c_left, c_right = st.columns([1, 2])

    with c_left:
        eps = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5, 0.1, key="db_eps")
        min_samples = st.slider("Min samples", 2, 30, 5, key="db_min")

    # Train
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels_db = db.fit_predict(X)
    n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    n_noise = (labels_db == -1).sum()

    sil_db = silhouette_score(X, labels_db) if n_clusters_db > 1 and n_clusters_db < len(X) else 0
    results["DBSCAN"] = {"labels": labels_db, "silhouette": sil_db, "k": n_clusters_db}

    with c_left:
        metric_card("Clusters Found", str(n_clusters_db), color="info")
        metric_card("Noise Points", f"{n_noise} ({n_noise/len(X)*100:.1f}%)", color="danger")
        if sil_db > 0:
            metric_card("Silhouette Score", f"{sil_db:.4f}", color="warning")

    with c_right:
        color_labels = labels_db.copy().astype(str)
        color_labels[labels_db == -1] = "Noise"

        if viz_mode == "3D (PCA)" and X_pca.shape[1] >= 3:
            fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                                color=color_labels, title=f"DBSCAN (eps={eps}, min_samples={min_samples})",
                                color_discrete_sequence=PALETTE + ["#555555"])
        else:
            fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=color_labels,
                             title=f"DBSCAN (eps={eps}, min_samples={min_samples})",
                             labels={"x": "PC1", "y": "PC2"},
                             color_discrete_sequence=PALETTE + ["#555555"])
        fig = apply_theme(fig)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    show_code(f"""from sklearn.cluster import DBSCAN

X = StandardScaler().fit_transform(df[{feature_cols}].dropna())
db = DBSCAN(eps={eps}, min_samples={min_samples})
labels = db.fit_predict(X)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"Clusters: {{n_clusters}}, Noise: {{n_noise}}")
""")

# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

with tab_compare:
    section_header("Method Comparison", "Side-by-side evaluation", "🏆")

    if results:
        comp_df = pd.DataFrame({
            name: {"Clusters": info["k"], "Silhouette": round(info["silhouette"], 4)}
            for name, info in results.items()
        }).T
        comp_df.index.name = "Method"
        st.dataframe(comp_df, use_container_width=True)

        comp_plot = comp_df.reset_index().rename(columns={"index": "Method"})
        fig = px.bar(comp_plot, x="Method", y="Silhouette",
                     color="Method", color_discrete_sequence=PALETTE,
                     title="Silhouette Score Comparison",
                     labels={"Silhouette": "Silhouette Score"})
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Side by side visualization
        st.markdown("#### Visual Comparison")
        cols = st.columns(len(results))
        for col, (name, info) in zip(cols, results.items()):
            with col:
                fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                                 color=info["labels"].astype(str),
                                 title=name, labels={"x": "PC1", "y": "PC2"},
                                 color_discrete_sequence=PALETTE)
                fig = apply_theme(fig)
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        log_to_report("Clustering", comp_df.to_html())
