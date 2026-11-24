
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="RetailMind Dashboard",
    page_icon="📦",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "reports" / "inventory_recommendations.csv"

df = pd.read_csv(DATA_PATH)

CLUSTER_PATH = BASE_DIR.parent / "reports" / "product_clustering_results.csv"

cluster_df = pd.read_csv(CLUSTER_PATH)

ASSOCIATION_PATH = BASE_DIR.parent / "reports" / "association_rules.csv"

association_df = pd.read_csv(ASSOCIATION_PATH)

FORECAST_PATH = BASE_DIR.parent / "reports" / "demand_predictions.csv"

forecast_df = pd.read_csv(FORECAST_PATH)

df = df.merge(
    forecast_df[
        ["Date", "Store ID", "Product ID", "Predicted Demand"]
    ],
    on=["Date", "Store ID", "Product ID"],
    how="left"
)


st.title("RetailMind")
st.subheader("Demand Forecasting & Inventory Intelligence")

st.write(
    "An intelligent retail decision-support system for demand analysis "
    "and inventory management."
)

st.divider()

st.subheader("Business KPIs")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Store-Product Combinations",
    len(df)
)

col2.metric(
    "Total Products",
    df["Product ID"].nunique()
)

col3.metric(
    "Total Stores",
    df["Store ID"].nunique()
)

reorder_count = (
    df["Inventory Status"] == "Reorder Required"
).sum()

healthy_count = (
    df["Inventory Status"] == "Healthy"
).sum()

risk_percent = round((reorder_count / len(df)) * 100, 2)

col4.metric(
    "Reorder Required",
    reorder_count
)

col5.metric(
    "Inventory Risk",
    f"{risk_percent}%"
)

st.info(
    f"{risk_percent}% of store-product combinations require replenishment."
)

st.divider()

st.subheader("Inventory Filters")

selected_store = st.selectbox(
    "Select Store",
    ["All Stores"] + sorted(df["Store ID"].unique().tolist())
)

if selected_store == "All Stores":
    filtered_df = df.copy()
else:
    filtered_df = df[df["Store ID"] == selected_store].copy()

st.subheader("Inventory Recommendations")

st.dataframe(
    filtered_df[
        [
            "Store ID",
            "Product ID",
            "Category",
            "Inventory Level",
            "Average Daily Demand",
            "Predicted Demand",
            "Stock Coverage",
            "Reorder Point",
            "Reorder Quantity",
            "Inventory Status"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

st.subheader("Model-Predicted Demand by Product")

prediction_data = (
    filtered_df.groupby("Product ID")["Predicted Demand"]
    .mean()
    .sort_values(ascending=False)
)

st.bar_chart(prediction_data)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Inventory Status Breakdown")

    status_counts = filtered_df["Inventory Status"].value_counts()

    st.bar_chart(status_counts)

with col2:
    st.subheader("Average Daily Demand by Product")

    demand_data = (
        filtered_df.groupby("Product ID")["Average Daily Demand"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(demand_data)

st.divider()

st.subheader("Reorder Recommendations")

reorder_df = filtered_df[
    filtered_df["Inventory Status"] == "Reorder Required"
].sort_values("Reorder Quantity", ascending=False)

st.dataframe(
    reorder_df[
        [
            "Store ID",
            "Product ID",
            "Inventory Level",
            "Stock Coverage",
            "Reorder Quantity"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

st.divider()

st.subheader("Demand Forecasting Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Mean Absolute Error (MAE)", "68.93")
col2.metric("Root Mean Squared Error (RMSE)", "88.17")
col3.metric("R² Score", "0.3333")

st.caption(
    "Performance measured on the held-out test dataset. "
    "Results are experimental and depend on dataset quality."
)
st.divider()

st.subheader("Product Clustering Analysis")

cluster_choice = st.selectbox(
    "Select Clustering Algorithm",
    ["Hierarchical Clustering", "DBSCAN"]
)

if cluster_choice == "Hierarchical Clustering":
    cluster_column = "Hierarchical Cluster"
else:
    cluster_column = "DBSCAN Cluster"

st.bar_chart(
    cluster_df[cluster_column].value_counts().sort_index()
)

st.dataframe(
    cluster_df[
        [
            "Store ID",
            "Product ID",
            cluster_column
        ]
    ],
    use_container_width=True,
    hide_index=True
)
st.divider()

st.subheader("Association Rule Mining")

st.write(
    "Discover products that are frequently purchased together "
    "using the Apriori algorithm."
)

st.metric(
    "Strong Association Rules",
    len(association_df)
)

st.dataframe(
    association_df[
        [
            "Antecedents",
            "Consequents",
            "support",
            "confidence",
            "lift"
        ]
    ].head(20),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")


st.caption(
    "RetailMind | Machine Learning-Based Demand Forecasting "
    "and Inventory Intelligence System"
)
