import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import apply_theme, section_header, show_code, log_to_report, PALETTE, check_data

st.set_page_config(page_title="Association Rules | DataMineHub", page_icon="🔗", layout="wide")

ok, df = check_data()
if not ok:
    st.stop()

section_header("Frequent Patterns & Association Rules", "Apriori algorithm, support, confidence, lift", "🔗")

# ─── Detect suitable columns ─────────────────────────────────────────────────

# For retail data: InvoiceNo + Description
# For generic data: user picks transaction ID col + item col
has_retail_cols = all(c in df.columns for c in ["InvoiceNo", "Description"])

with st.sidebar:
    st.markdown("### ⚙️ Association Rules Config")

    if has_retail_cols:
        st.success("Retail dataset detected!")
        trans_col = "InvoiceNo"
        item_col = "Description"
    else:
        all_cols = df.columns.tolist()
        trans_col = st.selectbox("Transaction ID column", all_cols, key="ar_trans")
        item_col = st.selectbox("Item / Product column", [c for c in all_cols if c != trans_col], key="ar_item")

    st.markdown("---")
    min_support = st.slider("Min Support", 0.005, 0.15, 0.02, 0.005, key="ar_sup",
                            help="Minimum fraction of transactions containing the itemset")
    min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05, key="ar_conf",
                               help="Minimum probability of consequent given antecedent")
    min_lift = st.slider("Min Lift", 0.5, 10.0, 1.0, 0.1, key="ar_lift",
                         help="Values > 1 indicate positive association")
    max_items = st.slider("Max items per rule", 2, 6, 3, key="ar_max")
    top_n_rules = st.slider("Top N rules to display", 5, 50, 15, key="ar_topn")

# ─── Build Basket ────────────────────────────────────────────────────────────

st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#6C63FF,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
section_header("Step 1: Build Transaction Baskets", "Group items by transaction", "🛒")

@st.cache_data
def build_baskets(data, t_col, i_col, max_len):
    """Build one-hot basket matrix from transaction data."""
    # Clean
    data = data[[t_col, i_col]].dropna()
    data[i_col] = data[i_col].astype(str).str.strip()
    data = data[data[i_col] != ""]

    # Keep only top items for performance
    top_items = data[i_col].value_counts().head(200).index
    data = data[data[i_col].isin(top_items)]

    basket = data.groupby([t_col, i_col]).size().unstack(fill_value=0)
    basket = (basket > 0).astype(bool)

    return basket

try:
    with st.spinner("Building baskets..."):
        basket = build_baskets(df, trans_col, item_col, max_items)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Transactions", f"{basket.shape[0]:,}")
    with c2:
        st.metric("Unique Items", f"{basket.shape[1]:,}")
    with c3:
        density = (basket.sum().sum()) / (basket.shape[0] * basket.shape[1]) * 100
        st.metric("Basket Density", f"{density:.2f}%")

    with st.expander("Preview basket (first 10 transactions)", expanded=False):
        st.dataframe(basket.head(10).astype(int), use_container_width=True)

    show_code(f"""import pandas as pd

# Build transaction baskets
basket = df.groupby(['{trans_col}', '{item_col}']).size().unstack(fill_value=0)
basket = (basket > 0).astype(bool)
print(f"Baskets: {{basket.shape[0]}} transactions × {{basket.shape[1]}} items")
""")

except Exception as e:
    st.error(f"Error building baskets: {e}")
    st.info("Make sure you selected the correct Transaction ID and Item columns.")
    st.stop()

# ─── Apriori ─────────────────────────────────────────────────────────────────

st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#FF6584,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
section_header("Step 2: Mine Frequent Itemsets (Apriori)", "Adjust min_support to find patterns", "⛏️")

try:
    from mlxtend.frequent_patterns import apriori, association_rules as gen_rules

    with st.spinner("Running Apriori algorithm..."):
        freq_items = apriori(basket, min_support=min_support, use_colnames=True, max_len=max_items)

    if len(freq_items) == 0:
        st.warning("No frequent itemsets found. Try lowering the min support slider.")
        st.stop()

    freq_items["length"] = freq_items["itemsets"].apply(len)
    st.success(f"Found **{len(freq_items)}** frequent itemsets")

    tab_table, tab_chart = st.tabs(["📋 Itemsets Table", "📊 Visualization"])

    with tab_table:
        display_fi = freq_items.copy()
        display_fi["itemsets"] = display_fi["itemsets"].apply(lambda x: ", ".join(list(x)))
        st.dataframe(display_fi.sort_values("support", ascending=False).head(50),
                     use_container_width=True, hide_index=True)

    with tab_chart:
        fi_by_len = freq_items.groupby("length").size().reset_index(name="count")
        fig = px.bar(fi_by_len, x="length", y="count", title="Itemsets by Length",
                     color="length", color_continuous_scale="Viridis",
                     labels={"length": "Itemset Size", "count": "Count"})
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    show_code(f"""from mlxtend.frequent_patterns import apriori

freq_items = apriori(basket, min_support={min_support},
                     use_colnames=True, max_len={max_items})
print(f"Found {{len(freq_items)}} frequent itemsets")
print(freq_items.sort_values('support', ascending=False).head(10))
""")

except ImportError:
    st.error("mlxtend not installed. Run: `pip install mlxtend`")
    st.stop()

# ─── Association Rules ───────────────────────────────────────────────────────

st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#2ED47A,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)
section_header("Step 3: Generate Association Rules", "Adjust confidence and lift thresholds", "📜")

sort_by = st.selectbox("Sort rules by", ["lift", "confidence", "support", "leverage", "conviction"], key="ar_sort")

try:
    rules = gen_rules(freq_items, metric="confidence", min_threshold=min_confidence)

    if len(rules) == 0:
        st.warning("No rules found. Try lowering the min confidence or min support.")
        st.stop()

    # Filter by lift
    rules = rules[rules["lift"] >= min_lift]

    if len(rules) == 0:
        st.warning("No rules meet the lift threshold. Try lowering it.")
        st.stop()

    rules = rules.sort_values(sort_by, ascending=False)

    st.success(f"Generated **{len(rules)}** association rules")

    # Display rules
    display_rules = rules.head(top_n_rules).copy()
    display_rules["antecedents"] = display_rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    display_rules["consequents"] = display_rules["consequents"].apply(lambda x: ", ".join(list(x)))

    tab_rules, tab_scatter, tab_topn = st.tabs(["📋 Rules Table", "📊 Scatter Plot", "🏆 Top Rules"])

    with tab_rules:
        st.dataframe(
            display_rules[["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]].round(4),
            use_container_width=True, hide_index=True
        )

    with tab_scatter:
        scatter_data = rules.head(200).copy()
        scatter_data["antecedents"] = scatter_data["antecedents"].apply(lambda x: ", ".join(list(x)))
        scatter_data["consequents"] = scatter_data["consequents"].apply(lambda x: ", ".join(list(x)))
        scatter_data["rule"] = scatter_data["antecedents"] + " → " + scatter_data["consequents"]

        fig = px.scatter(scatter_data, x="support", y="confidence",
                         size="lift", color="lift",
                         color_continuous_scale="Plasma",
                         hover_data=["rule", "lift"],
                         title="Support vs Confidence (size & color = Lift)",
                         labels={"support": "Support", "confidence": "Confidence"})
        fig = apply_theme(fig)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab_topn:
        top_rules = display_rules.head(top_n_rules)
        top_rules["rule"] = top_rules["antecedents"] + " → " + top_rules["consequents"]
        fig = px.bar(top_rules, x="lift", y="rule", orientation="h",
                     color="confidence", color_continuous_scale="Viridis",
                     title=f"Top {min(top_n_rules, len(top_rules))} Rules by Lift",
                     labels={"lift": "Lift", "rule": "", "confidence": "Confidence"})
        fig = apply_theme(fig)
        fig.update_layout(height=max(400, top_n_rules * 30), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    show_code(f"""from mlxtend.frequent_patterns import association_rules

rules = association_rules(freq_items, metric='confidence',
                          min_threshold={min_confidence})
rules = rules[rules['lift'] >= {min_lift}]
rules = rules.sort_values('{sort_by}', ascending=False)
print(f"Generated {{len(rules)}} rules")
print(rules[['antecedents','consequents','support','confidence','lift']].head({top_n_rules}))
""")

    log_to_report("Association Rules", f"""
    <p>Parameters: min_support={min_support}, min_confidence={min_confidence}, min_lift={min_lift}</p>
    <p>Found {len(freq_items)} frequent itemsets, {len(rules)} association rules.</p>
    {display_rules[['antecedents','consequents','support','confidence','lift']].head(10).to_html(index=False)}
    """)

except Exception as e:
    st.error(f"Error generating rules: {e}")
