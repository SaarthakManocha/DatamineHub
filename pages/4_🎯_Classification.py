import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.helpers import apply_theme, section_header, metric_card, show_code, log_to_report, PALETTE, check_data
from utils.data_loader import get_rfm

st.set_page_config(page_title="Classification | DataMineHub", page_icon="🎯", layout="wide")

section_header("Classification", "Decision Tree, Naive Bayes, SVM, KNN, Ensembles & model comparison", "🎯")

# ─── Data Source Selection ────────────────────────────────────────────────────

ok, raw_df = check_data()
if not ok:
    st.stop()

rfm = get_rfm()

with st.sidebar:
    st.markdown("### ⚙️ Classification Config")

    data_source = st.radio("Data source", ["Uploaded Dataset", "RFM Features (from retail data)"],
                           index=1 if rfm is not None else 0, key="cls_source")

if data_source == "RFM Features (from retail data)" and rfm is not None:
    df = rfm.drop(columns=["CustomerID"], errors="ignore")
else:
    df = raw_df.copy()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

if len(numeric_cols) < 1:
    st.warning("Need at least 1 numeric column for classification.")
    st.stop()

# ─── Feature / Target Selection ──────────────────────────────────────────────

st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#6C63FF,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
section_header("Step 1: Select Features & Target", "Choose what to predict", "🎛️")

with st.sidebar:
    # Smart default: if 'HighValue' exists, use it as target
    default_target = "HighValue" if "HighValue" in all_cols else all_cols[-1]
    target_col = st.selectbox("Target column (y)", all_cols,
                              index=all_cols.index(default_target) if default_target in all_cols else 0,
                              key="cls_target")

    available_features = [c for c in numeric_cols if c != target_col]
    feature_cols = st.multiselect("Feature columns (X)", available_features,
                                  default=available_features[:6] if len(available_features) >= 6 else available_features,
                                  key="cls_features")

    st.markdown("---")
    test_size = st.slider("Test split ratio", 0.1, 0.5, 0.25, 0.05, key="cls_split")
    random_state = st.number_input("Random seed", 0, 999, 42, key="cls_seed")

if not feature_cols:
    st.info("👈 Select feature columns from the sidebar.")
    st.stop()

# Prepare data
X = df[feature_cols].dropna()
y = df.loc[X.index, target_col]

# Handle non-numeric target
if not pd.api.types.is_numeric_dtype(y):
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
    class_names = list(le.classes_)
else:
    y = y.dropna()
    X = X.loc[y.index]
    class_names = [str(c) for c in sorted(y.unique())]

# Drop any remaining NaN
mask = X.notna().all(axis=1) & y.notna()
X, y = X[mask], y[mask]

if len(X) < 20:
    st.warning("Not enough clean data points for classification. Try different columns or clean the data first.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Samples", f"{len(X):,}")
with c2:
    st.metric("Features", f"{len(feature_cols)}")
with c3:
    st.metric("Classes", f"{y.nunique()}")

# ─── Algorithm Selection ─────────────────────────────────────────────────────

st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#FF6584,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
section_header("Step 2: Select & Configure Algorithms", "Check any combination to compare", "🧠")

col_algo1, col_algo2, col_algo3 = st.columns(3)

models = {}

with col_algo1:
    if st.checkbox("🌳 Decision Tree", value=True, key="cls_dt"):
        dt_depth = st.slider("Max depth", 1, 20, 5, key="cls_dt_depth")
        dt_crit = st.selectbox("Criterion", ["gini", "entropy"], key="cls_dt_crit")
        models["Decision Tree"] = {
            "model": DecisionTreeClassifier(max_depth=dt_depth, criterion=dt_crit, random_state=random_state),
            "code": f"DecisionTreeClassifier(max_depth={dt_depth}, criterion='{dt_crit}')",
        }

    if st.checkbox("📊 Naive Bayes", value=True, key="cls_nb"):
        models["Naive Bayes"] = {
            "model": GaussianNB(),
            "code": "GaussianNB()",
        }

    if st.checkbox("🚀 AdaBoost", value=False, key="cls_ada"):
        ada_n = st.slider("N estimators (Ada)", 10, 200, 50, 10, key="cls_ada_n")
        models["AdaBoost"] = {
            "model": AdaBoostClassifier(n_estimators=ada_n, random_state=random_state, algorithm="SAMME"),
            "code": f"AdaBoostClassifier(n_estimators={ada_n})",
        }

with col_algo2:
    if st.checkbox("🌲 Random Forest", value=True, key="cls_rf"):
        rf_n = st.slider("N estimators (RF)", 10, 300, 100, 10, key="cls_rf_n")
        rf_depth = st.slider("Max depth (RF)", 1, 30, 10, key="cls_rf_depth")
        models["Random Forest"] = {
            "model": RandomForestClassifier(n_estimators=rf_n, max_depth=rf_depth, random_state=random_state),
            "code": f"RandomForestClassifier(n_estimators={rf_n}, max_depth={rf_depth})",
        }

    if st.checkbox("📈 Gradient Boosting", value=False, key="cls_gb"):
        gb_n = st.slider("N estimators (GB)", 10, 300, 100, 10, key="cls_gb_n")
        gb_lr = st.slider("Learning rate (GB)", 0.01, 1.0, 0.1, 0.01, key="cls_gb_lr")
        models["Gradient Boosting"] = {
            "model": GradientBoostingClassifier(n_estimators=gb_n, learning_rate=gb_lr, random_state=random_state),
            "code": f"GradientBoostingClassifier(n_estimators={gb_n}, learning_rate={gb_lr})",
        }

with col_algo3:
    if st.checkbox("🔮 SVM", value=False, key="cls_svm"):
        svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="cls_svm_k")
        svm_c = st.slider("C (regularization)", 0.1, 10.0, 1.0, 0.1, key="cls_svm_c")
        models["SVM"] = {
            "model": SVC(kernel=svm_kernel, C=svm_c, probability=True, random_state=random_state),
            "code": f"SVC(kernel='{svm_kernel}', C={svm_c}, probability=True)",
        }

    if st.checkbox("👥 KNN", value=False, key="cls_knn"):
        knn_k = st.slider("K neighbors", 1, 25, 5, key="cls_knn_k")
        knn_metric = st.selectbox("Distance", ["euclidean", "manhattan", "minkowski"], key="cls_knn_m")
        models["KNN"] = {
            "model": KNeighborsClassifier(n_neighbors=knn_k, metric=knn_metric),
            "code": f"KNeighborsClassifier(n_neighbors={knn_k}, metric='{knn_metric}')",
        }

if not models:
    st.info("☝️ Select at least one algorithm above.")
    st.stop()

# ─── Train Models ────────────────────────────────────────────────────────────

st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#2ED47A,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)

if st.button("🚀 Train All Selected Models", use_container_width=True, type="primary"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() <= 20 else None)

    # Scale features
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    results = {}
    trained_models = {}

    progress = st.progress(0)
    for i, (name, info) in enumerate(models.items()):
        with st.spinner(f"Training {name}..."):
            model = info["model"]
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            avg = "binary" if y.nunique() == 2 else "weighted"
            results[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
                "Recall": recall_score(y_test, y_pred, average=avg, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average=avg, zero_division=0),
            }
            trained_models[name] = {
                "model": model,
                "y_pred": y_pred,
                "y_test": y_test,
                "X_test": X_test_s,
                "code": info["code"],
            }

        progress.progress((i + 1) / len(models))

    # ─── Results ──────────────────────────────────────────────────────────

    section_header("Step 3: Results", "Confusion matrices, ROC curves & model comparison", "🏆")

    # Leaderboard
    results_df = pd.DataFrame(results).T.round(4)
    results_df = results_df.sort_values("F1 Score", ascending=False)
    results_df.index.name = "Model"

    st.markdown("### 🏆 Model Leaderboard")

    fig = px.bar(results_df.reset_index().melt(id_vars="Model"),
                 x="Model", y="value", color="variable", barmode="group",
                 title="Model Comparison", color_discrete_sequence=PALETTE,
                 labels={"value": "Score", "variable": "Metric"})
    fig = apply_theme(fig)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(results_df, use_container_width=True)

    # Per-model details
    st.markdown("### 📋 Per-Model Details")
    model_tabs = st.tabs(list(trained_models.keys()))

    for tab, (name, info) in zip(model_tabs, trained_models.items()):
        with tab:
            y_pred = info["y_pred"]
            y_test_m = info["y_test"]

            c1, c2 = st.columns(2)
            with c1:
                # Confusion matrix
                cm = confusion_matrix(y_test_m, y_pred)
                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                title=f"Confusion Matrix — {name}",
                                labels={"x": "Predicted", "y": "Actual"})
                fig = apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                # ROC curve (binary only)
                if y.nunique() == 2 and hasattr(info["model"], "predict_proba"):
                    try:
                        y_proba = info["model"].predict_proba(info["X_test"])[:, 1]
                        fpr, tpr, _ = roc_curve(y_test_m, y_proba)
                        roc_auc = auc(fpr, tpr)
                        fig = px.area(x=fpr, y=tpr, title=f"ROC Curve — {name} (AUC={roc_auc:.3f})",
                                      labels={"x": "False Positive Rate", "y": "True Positive Rate"})
                        fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="#8B8D97"))
                        fig.update_traces(fillcolor=PALETTE[0] + "33", line_color=PALETTE[0])
                        fig = apply_theme(fig)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.info("ROC curve not available for this model configuration.")
                else:
                    # Classification report
                    report = classification_report(y_test_m, y_pred, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).T.round(4)
                    st.dataframe(report_df, use_container_width=True)

            # Feature importance (tree-based)
            if hasattr(info["model"], "feature_importances_"):
                imp = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": info["model"].feature_importances_
                }).sort_values("Importance", ascending=True)
                fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                             title=f"Feature Importance — {name}",
                             color="Importance", color_continuous_scale="Viridis")
                fig = apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            # Decision tree text
            if name == "Decision Tree":
                with st.expander("🌳 Tree Structure (text)"):
                    tree_text = export_text(info["model"], feature_names=feature_cols, max_depth=5)
                    st.code(tree_text)

            # Code for this model
            show_code(f"""from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Prepare data
X = df[{feature_cols}]
y = df['{target_col}']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state={random_state})

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train {name}
from sklearn.{info['model'].__module__.split('.')[-1]} import {type(info['model']).__name__}
model = {info['code']}
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {{accuracy_score(y_test, y_pred):.4f}}")
print(classification_report(y_test, y_pred))
""")

    # Report log
    log_to_report("Classification",
                  f"<p>Target: {target_col}, Features: {feature_cols}, Split: {test_size}</p>" +
                  results_df.to_html())
