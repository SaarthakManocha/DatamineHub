import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os, urllib.request

DATASET_URL = "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv"
LOCAL_PATH = "data/online_retail.csv"


def _ensure_dataset():
    """Download default dataset if not already present."""
    if not os.path.exists(LOCAL_PATH):
        os.makedirs("data", exist_ok=True)
        with st.spinner("Downloading default dataset (~45 MB)…"):
            urllib.request.urlretrieve(DATASET_URL, LOCAL_PATH)


@st.cache_data
def load_default_dataset():
    """Load the Online Retail dataset from the data/ folder."""
    _ensure_dataset()
    try:
        df = pd.read_csv(LOCAL_PATH, encoding="latin-1")
    except FileNotFoundError:
        try:
            df = pd.read_excel("data/online_retail.xlsx")
        except FileNotFoundError:
            return None

    # Basic cleaning
    df.columns = df.columns.str.strip()

    # Standardize column names
    col_map = {}
    for col in df.columns:
        lower = col.lower().replace(" ")
        if "invoice" in lower and "date" not in lower and "no" not in lower:
            col_map[col] = "InvoiceNo"
        elif "invoicedate" in lower or ("invoice" in lower and "date" in lower):
            col_map[col] = "InvoiceDate"
        elif "stock" in lower:
            col_map[col] = "StockCode"
        elif "description" in lower:
            col_map[col] = "Description"
        elif "quantity" in lower:
            col_map[col] = "Quantity"
        elif "price" in lower or "unitprice" in lower:
            col_map[col] = "UnitPrice"
        elif "customer" in lower:
            col_map[col] = "CustomerID"
        elif "country" in lower:
            col_map[col] = "Country"

    df = df.rename(columns=col_map)
    return df


@st.cache_data
def preprocess_retail(df):
    """Clean the retail dataset and prepare for analysis."""
    cleaned = df.copy()

    # Drop rows with missing CustomerID
    if "CustomerID" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["CustomerID"])
        cleaned["CustomerID"] = cleaned["CustomerID"].astype(int).astype(str)

    # Remove cancelled orders (InvoiceNo starting with 'C')
    if "InvoiceNo" in cleaned.columns:
        cleaned["InvoiceNo"] = cleaned["InvoiceNo"].astype(str)
        cleaned = cleaned[~cleaned["InvoiceNo"].str.startswith("C")]

    # Remove zero/negative quantities and prices
    if "Quantity" in cleaned.columns:
        cleaned = cleaned[cleaned["Quantity"] > 0]
    if "UnitPrice" in cleaned.columns:
        cleaned = cleaned[cleaned["UnitPrice"] > 0]

    # Parse dates
    if "InvoiceDate" in cleaned.columns:
        cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")

    # Create TotalPrice
    if "Quantity" in cleaned.columns and "UnitPrice" in cleaned.columns:
        cleaned["TotalPrice"] = cleaned["Quantity"] * cleaned["UnitPrice"]

    return cleaned


@st.cache_data
def derive_rfm(df):
    """Derive RFM (Recency, Frequency, Monetary) features at customer level."""
    if not all(c in df.columns for c in ["CustomerID", "InvoiceNo", "InvoiceDate", "TotalPrice"]):
        return None

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    # Additional derived features
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    rfm["AvgOrderValue"] = rfm["AvgOrderValue"].replace([np.inf, -np.inf], 0)

    # Binary target: High-Value Customer (top 25% by Monetary)
    threshold = rfm["Monetary"].quantile(0.75)
    rfm["HighValue"] = (rfm["Monetary"] >= threshold).astype(int)

    return rfm


def get_data():
    """Get data from session state."""
    if "dataset" in st.session_state and st.session_state["dataset"] is not None:
        return st.session_state["dataset"]
    return None


def get_rfm():
    """Get RFM data from session state."""
    if "rfm_data" in st.session_state and st.session_state["rfm_data"] is not None:
        return st.session_state["rfm_data"]
    return None


def get_cleaned():
    """Get cleaned data from session state."""
    if "cleaned_data" in st.session_state and st.session_state["cleaned_data"] is not None:
        return st.session_state["cleaned_data"]
    return None
