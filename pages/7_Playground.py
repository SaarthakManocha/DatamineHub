import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from utils.helpers import apply_theme, section_header, metric_card, show_code, PALETTE

st.set_page_config(page_title="Playground | DataMineHub",  layout="wide")

section_header("Synthetic Dataset Playground", "Generate custom 2D shapes and stress-test algorithms")

st.markdown("""
<div style="background:linear-gradient(135deg,#6C63FF11,#FF658411);border:1px solid #2A2D3A;border-radius:12px;padding:1.2rem;margin-bottom:1.5rem;">
    <div style="color:#FAFAFA;font-weight:600;margin-bottom:0.3rem;">Why this matters</div>
    <div style="color:#8B8D97;font-size:0.9rem;">
        Different algorithms have different strengths. K-Means works great on spherical blobs but fails on crescent moons.
        DBSCAN handles arbitrary shapes but struggles with varying densities. <strong>See it for yourself!</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Dataset Generation ---
with st.sidebar:
    st.markdown("### Generate Dataset")
    shape = st.selectbox("Shape", ["Blobs", "Moons", "Circles", "Anisotropic", "Varied Density"], key="pg_shape")
    n_samples = st.slider("Number of samples", 100, 2000, 500, 50, key="pg_n")
    noise = st.slider("Noise level", 0.01, 0.5, 0.08, 0.01, key="pg_noise")
    random_seed = st.slider("Random seed", 0, 100, 42, key="pg_seed")

    if shape == "Blobs":
        n_centers = st.slider("Number of centers", 2, 8, 3, key="pg_centers")

    st.markdown("---")
    st.markdown("### Algorithm Mode")
    mode = st.radio("Task", ["Clustering", "Classification"], key="pg_mode")

# Generate data
np.random.seed(random_seed)

if shape == "Blobs":
    X, y_true = make_blobs(n_samples=n_samples, centers=n_centers, cluster_std=noise * 10, random_state=random_seed)
elif shape == "Moons":
    X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=random_seed)
elif shape == "Circles":
    X, y_true = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_seed)
elif shape == "Anisotropic":
    X, y_true = make_blobs(n_samples=n_samples, centers=3, random_state=random_seed)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X = np.dot(X, transformation)
else: # Varied Density
    X1, y1 = make_blobs(n_samples=n_samples // 3, centers=[[0, 0]], cluster_std=0.3, random_state=random_seed)
    X2, y2 = make_blobs(n_samples=n_samples // 3, centers=[[3, 3]], cluster_std=1.0, random_state=random_seed)
    X3, y3 = make_blobs(n_samples=n_samples // 3, centers=[[-3, 3]], cluster_std=0.5, random_state=random_seed)
    X = np.vstack([X1, X2, X3])
    y_true = np.hstack([np.zeros(len(y1)), np.ones(len(y2)), np.full(len(y3), 2)]).astype(int)

# --- Show Generated Data ---
st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#6C63FF,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
section_header("Generated Dataset", f"{shape} - {n_samples} samples, noise={noise}")

c1, c2 = st.columns([2, 1])
with c1:
    fig = px.scatter(x=X[:, 0], y=X[:, 1], color=y_true.astype(str),
                     title=f"Ground Truth - {shape}",
                     labels={"x": "Feature 1", "y": "Feature 2", "color": "True Label"},
                     color_discrete_sequence=PALETTE)
    fig = apply_theme(fig)
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    metric_card("Samples", str(n_samples), color="primary")
    metric_card("True Classes", str(len(np.unique(y_true))), color="success")
    metric_card("Shape", shape, color="info")

    show_code(f"""from sklearn.datasets import make_{shape.lower() if shape in ['Blobs','Moons','Circles'] else 'blobs'}

X, y = make_{shape.lower() if shape in ['Blobs','Moons','Circles'] else 'blobs'}(
    n_samples={n_samples}, {'noise' if shape != 'Blobs' else 'cluster_std'}={noise if shape != 'Blobs' else noise*10},
    random_state={random_seed}
)
""")

# --- Apply Algorithms ---
st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#FF6584,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)

if mode == "Clustering":
    section_header("Apply Clustering Algorithms", "See how each algorithm handles this shape")

    algos = st.multiselect("Select algorithms", ["K-Means", "Hierarchical", "DBSCAN"],
                           default=["K-Means", "DBSCAN"], key="pg_cl_algos")

    params = {}
    param_cols = st.columns(len(algos)) if algos else []

    for col, algo in zip(param_cols, algos):
        with col:
            st.markdown(f"**{algo}**")
            if algo == "K-Means":
                params[algo] = {"k": st.slider("K", 2, 10, len(np.unique(y_true)), key="pg_km_k")}
            elif algo == "Hierarchical":
                params[algo] = {
                    "k": st.slider("N clusters", 2, 10, len(np.unique(y_true)), key="pg_h_k"),
                    "linkage": st.selectbox("Linkage", ["ward", "complete", "average"], key="pg_h_l"),
                }
            elif algo == "DBSCAN":
                params[algo] = {
                    "eps": st.slider("Epsilon", 0.05, 3.0, 0.3, 0.05, key="pg_db_e"),
                    "min_samples": st.slider("Min samples", 2, 20, 5, key="pg_db_m"),
                }

    if algos and st.button(" Run Clustering", use_container_width=True, type="primary"):
        X_scaled = StandardScaler().fit_transform(X)
        result_cols = st.columns(len(algos))

        for col, algo in zip(result_cols, algos):
            with col:
                if algo == "K-Means":
                    model = KMeans(n_clusters=params[algo]["k"], n_init=10, random_state=42)
                    labels = model.fit_predict(X_scaled)
                elif algo == "Hierarchical":
                    model = AgglomerativeClustering(n_clusters=params[algo]["k"], linkage=params[algo]["linkage"])
                    labels = model.fit_predict(X_scaled)
                elif algo == "DBSCAN":
                    model = DBSCAN(eps=params[algo]["eps"], min_samples=params[algo]["min_samples"])
                    labels = model.fit_predict(X_scaled)

                color_labels = labels.astype(str)
                if -1 in labels:
                    color_labels[labels == -1] = "Noise"

                n_cl = len(set(labels)) - (1 if -1 in labels else 0)
                sil = silhouette_score(X_scaled, labels) if n_cl > 1 and n_cl < len(X) else 0

                fig = px.scatter(x=X[:, 0], y=X[:, 1], color=color_labels,
                                 title=f"{algo} (Sil: {sil:.3f})",
                                 labels={"x": "F1", "y": "F2"},
                                 color_discrete_sequence=PALETTE + ["#555555"])
                fig = apply_theme(fig)
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Clusters: {n_cl} | Silhouette: {sil:.4f}")

else: # Classification
    section_header("Apply Classification Algorithms", "Train on this synthetic data")

    algos = st.multiselect("Select algorithms",
                           ["Decision Tree", "Naive Bayes", "Random Forest", "SVM", "KNN"],
                           default=["Decision Tree", "SVM", "KNN"], key="pg_cls_algos")

    test_frac = st.slider("Test split", 0.1, 0.5, 0.3, 0.05, key="pg_cls_split")

    if algos and st.button(" Run Classification", use_container_width=True, type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=test_frac, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        result_cols = st.columns(len(algos))
        results = {}

        for col, algo in zip(result_cols, algos):
            with col:
                if algo == "Decision Tree":
                    model = DecisionTreeClassifier(max_depth=5, random_state=42)
                elif algo == "Naive Bayes":
                    model = GaussianNB()
                elif algo == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif algo == "SVM":
                    model = SVC(kernel="rbf", random_state=42)
                elif algo == "KNN":
                    model = KNeighborsClassifier(n_neighbors=5)

                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
                acc = accuracy_score(y_test, y_pred)
                results[algo] = acc

                # Decision boundary visualization
                h = 0.3
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
                Z = Z.reshape(xx.shape)

                fig = px.scatter(x=X_test[:, 0], y=X_test[:, 1], color=y_pred.astype(str),
                                 title=f"{algo} (Acc: {acc:.1%})",
                                 labels={"x": "F1", "y": "F2"},
                                 color_discrete_sequence=PALETTE)
                fig.add_trace(px.imshow(Z, x=np.arange(x_min, x_max, h),
                                        y=np.arange(y_min, y_max, h),
                                        color_continuous_scale="Viridis", opacity=0.15).data[0])
                fig = apply_theme(fig)
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Accuracy: {acc:.4f}")

        # Comparison
        if len(results) > 1:
            st.markdown("### Accuracy Comparison")
            fig = px.bar(x=list(results.keys()), y=list(results.values()),
                         color=list(results.keys()), color_discrete_sequence=PALETTE,
                         title="Algorithm Accuracy on Synthetic Data",
                         labels={"x": "Algorithm", "y": "Accuracy"})
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

# --- Key Takeaways ---
st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#2ED47A,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
section_header(" Key Insights", "What each shape teaches us about algorithms")

insights = {
    "Blobs": "**Spherical clusters** - K-Means excels here since it assumes spherical, equal-sized clusters. DBSCAN works too if you set eps correctly. This is the 'easy case' for most algorithms.",
    "Moons": "**Crescent shapes** - K-Means fails completely because it can't handle non-convex shapes. DBSCAN shines here because it follows density. Hierarchical with 'single' linkage can also work.",
    "Circles": "**Concentric rings** - Another case where K-Means fails. Density-based methods (DBSCAN) handle this well. For classification, SVM with RBF kernel can learn the circular boundary.",
    "Anisotropic": "**Stretched/elongated clusters** - K-Means struggles because clusters aren't spherical. Hierarchical with 'ward' linkage handles elongated shapes better.",
    "Varied Density": "**Different cluster densities** - DBSCAN struggles here because a single eps can't fit all densities. K-Means is surprisingly more robust since it just cares about centroids.",
}

st.markdown(f"""
<div style="background:#1A1D29;border-radius:12px;padding:1.5rem;border-left:3px solid #2ED47A;">
{insights.get(shape, "Try different shapes to see how algorithms behave!")}
</div>
""", unsafe_allow_html=True)
