import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(
    page_title="RetailMind",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR.parent / "reports"

NAVY = "#14213D"
TEAL = "#0D9488"
BLUE = "#2563EB"
AMBER = "#F59E0B"
RED = "#EF4444"
MUTED = "#64748B"
LIGHT_BG = "#F4F7FB"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {LIGHT_BG};
        color: #1F2937;
    }}

    [data-testid="stHeader"] {{
        background: rgba(244, 247, 251, 0.95);
    }}

    [data-testid="stSidebar"] {{
        background: {NAVY};
    }}

    [data-testid="stSidebar"] * {{
        color: #F8FAFC;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] .stCaption {{
        color: #CBD5E1 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        color: #F8FAFC !important;
    }}

    h1, h2, h3, h4 {{
        color: {NAVY} !important;
        font-weight: 700 !important;
    }}

    p, label, [data-testid="stCaptionContainer"] {{
        color: #475569;
    }}

    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }}

    [data-testid="stMetric"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 8px 2px 12px 2px !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {NAVY} !important;
        font-size: 2rem !important;
        font-weight: 750 !important;
    }}

    [data-testid="stMetricDelta"] {{
        color: {TEAL} !important;
    }}

    div[data-baseweb="select"] > div {{
        background: #FFFFFF;
        border: 1px solid #D7E0EA;
        border-radius: 9px;
    }}

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {{
        color: #1F2937 !important;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
    }}

    [data-testid="stDownloadButton"] button,
    .stButton button {{
        background: {TEAL};
        color: #FFFFFF;
        border: 1px solid {TEAL};
        border-radius: 8px;
        font-weight: 600;
    }}

    [data-testid="stDownloadButton"] button:hover,
    .stButton button:hover {{
        background: #0F766E;
        color: #FFFFFF;
        border-color: #0F766E;
    }}

    [data-testid="stAlert"] {{
        border-radius: 9px;
    }}

    hr {{
        border-color: #DCE4EE;
    }}

    [data-testid="stSlider"] {{
        padding-top: 0.4rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_data():
    inventory_path = REPORTS_DIR / "inventory_recommendations.csv"
    cluster_path = REPORTS_DIR / "product_clustering_results.csv"
    association_path = REPORTS_DIR / "association_rules.csv"
    forecast_path = REPORTS_DIR / "demand_predictions.csv"

    required_files = [
        inventory_path,
        cluster_path,
        association_path,
        forecast_path
    ]

    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "The following report files could not be found:\n"
            + "\n".join(missing_files)
        )

    inventory = pd.read_csv(inventory_path)
    clusters = pd.read_csv(cluster_path)
    associations = pd.read_csv(association_path)
    forecasts = pd.read_csv(forecast_path)

    required_inventory_columns = [
        "Store ID",
        "Product ID",
        "Inventory Status",
        "Average Daily Demand",
        "Inventory Level",
        "Stock Coverage",
        "Reorder Quantity"
    ]

    missing_columns = [
        column for column in required_inventory_columns
        if column not in inventory.columns
    ]

    if missing_columns:
        raise ValueError(
            "The inventory report is missing these columns: "
            + ", ".join(missing_columns)
        )

    forecast_columns = [
        "Date",
        "Store ID",
        "Product ID",
        "Predicted Demand"
    ]

    if "Predicted Demand" not in inventory.columns:
        if all(column in forecasts.columns for column in forecast_columns):
            inventory = inventory.merge(
                forecasts[forecast_columns].drop_duplicates(
                    subset=["Date", "Store ID", "Product ID"]
                ),
                on=["Date", "Store ID", "Product ID"],
                how="left"
            )
        else:
            inventory["Predicted Demand"] = pd.NA

    return inventory, clusters, associations


try:
    df, cluster_df, association_df = load_data()
except Exception as error:
    st.error(f"Unable to load RetailMind data: {error}")
    st.info(
        "Check that the reports folder is present at the project root "
        "and contains the required CSV files."
    )
    st.stop()


with st.sidebar:
    st.markdown("# 📦 RetailMind")
    st.caption("RETAIL INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Inventory",
            "Forecasting",
            "Clustering",
            "Association Rules"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.caption("Machine Learning-Based Retail Analytics")
    st.caption("Decision support for smarter inventory planning")


st.title("RetailMind")
st.caption("Demand Forecasting & Inventory Intelligence System")
st.divider()

st.subheader("Dashboard Filters")

filter1, filter2, filter3 = st.columns(3)

stores = ["All Stores"] + sorted(
    df["Store ID"].dropna().astype(str).unique().tolist()
)

if "Category" in df.columns:
    categories = ["All Categories"] + sorted(
        df["Category"].dropna().astype(str).unique().tolist()
    )
else:
    categories = ["All Categories"]

statuses = [
    "All Statuses",
    "Reorder Required",
    "Healthy"
]

with filter1:
    selected_store = st.selectbox("Select Store", stores)

with filter2:
    selected_category = st.selectbox("Select Category", categories)

with filter3:
    selected_status = st.selectbox("Inventory Status", statuses)


filtered_df = df.copy()

if selected_store != "All Stores":
    filtered_df = filtered_df[
        filtered_df["Store ID"].astype(str) == selected_store
    ]

if selected_category != "All Categories" and "Category" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["Category"].astype(str) == selected_category
    ]

if selected_status != "All Statuses":
    filtered_df = filtered_df[
        filtered_df["Inventory Status"] == selected_status
    ]

st.divider()

total_records = len(filtered_df)
total_products = filtered_df["Product ID"].nunique()
total_stores = filtered_df["Store ID"].nunique()

status_counts = (
    filtered_df["Inventory Status"]
    .value_counts()
    .reindex(
        ["Reorder Required", "Healthy"],
        fill_value=0
    )
)

reorder_count = int(status_counts["Reorder Required"])
healthy_count = int(status_counts["Healthy"])

risk_percent = (
    round((reorder_count / total_records) * 100, 1)
    if total_records > 0
    else 0
)


def make_bar_chart(data, x_field, y_field, color, horizontal=False):
    chart = alt.Chart(data).mark_bar(
        color=color,
        cornerRadiusEnd=5
    ).encode(
        x=alt.X(
            f"{y_field}:Q",
            title=None,
            axis=alt.Axis(
                labelColor=NAVY,
                titleColor=NAVY,
                gridColor="#E5EAF1"
            )
        ) if horizontal else alt.X(
            f"{x_field}:N",
            title=None,
            sort="-y",
            axis=alt.Axis(
                labelAngle=-35,
                labelColor=NAVY,
                titleColor=NAVY,
                grid=False
            )
        ),
        y=alt.Y(
            f"{x_field}:N",
            title=None,
            sort="-x",
            axis=alt.Axis(
                labelColor=NAVY,
                titleColor=NAVY,
                grid=False
            )
        ) if horizontal else alt.Y(
            f"{y_field}:Q",
            title=None,
            axis=alt.Axis(
                labelColor=NAVY,
                titleColor=NAVY,
                gridColor="#E5EAF1"
            )
        ),
        tooltip=[x_field, y_field]
    ).properties(
        height=300,
        background=LIGHT_BG
    ).configure_view(
        strokeOpacity=0
    ).configure_axis(
        domainColor="#CBD5E1",
        tickColor="#CBD5E1"
    )

    st.altair_chart(chart, use_container_width=True)


if page == "Overview":
    st.subheader("Business Overview")
    st.caption("A snapshot of inventory health and demand across your selected data.")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Records", f"{total_records:,}")
    c2.metric("Products", f"{total_products:,}")
    c3.metric("Stores", f"{total_stores:,}")
    c4.metric("Reorder Required", f"{reorder_count:,}")
    c5.metric("Inventory Risk", f"{risk_percent}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Inventory Status")

        status_chart = pd.DataFrame({
            "Status": ["Reorder Required", "Healthy"],
            "Count": [reorder_count, healthy_count]
        })

        status_colors = alt.Scale(
            domain=["Reorder Required", "Healthy"],
            range=[AMBER, TEAL]
        )

        status_plot = alt.Chart(status_chart).mark_bar(
            cornerRadiusEnd=5
        ).encode(
            x=alt.X(
                "Count:Q",
                title=None,
                axis=alt.Axis(
                    labelColor=NAVY,
                    gridColor="#E5EAF1"
                )
            ),
            y=alt.Y(
                "Status:N",
                sort=None,
                title=None,
                axis=alt.Axis(labelColor=NAVY)
            ),
            color=alt.Color(
                "Status:N",
                scale=status_colors,
                legend=None
            ),
            tooltip=["Status", "Count"]
        ).properties(
            height=260,
            background=LIGHT_BG
        ).configure_view(strokeOpacity=0)

        st.altair_chart(status_plot, use_container_width=True)

    with right:
        st.subheader("Average Demand by Product")

        if not filtered_df.empty:
            demand_data = (
                filtered_df.groupby("Product ID", as_index=False)[
                    "Average Daily Demand"
                ]
                .mean()
                .sort_values("Average Daily Demand", ascending=False)
                .head(10)
            )

            make_bar_chart(
                demand_data,
                "Product ID",
                "Average Daily Demand",
                BLUE
            )
        else:
            st.info("No demand data is available for these filters.")

    st.divider()
    st.subheader("Top Reorder Priorities")

    priority_df = filtered_df[
        filtered_df["Inventory Status"] == "Reorder Required"
    ].sort_values("Reorder Quantity", ascending=False)

    priority_columns = [
        "Store ID",
        "Product ID",
        "Inventory Level",
        "Stock Coverage",
        "Reorder Quantity"
    ]

    if not priority_df.empty:
        st.dataframe(
            priority_df[priority_columns].head(10).round(2),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No products require reordering in this selection.")

elif page == "Inventory":
    st.subheader("📦 Inventory Intelligence")
    st.caption("Monitor stock levels and identify replenishment requirements.")

    c1, c2 = st.columns(2)
    c1.metric("Reorder Required", f"{reorder_count:,}")
    c2.metric("Healthy", f"{healthy_count:,}")

    st.divider()
    st.subheader("Inventory Recommendations")

    inventory_columns = [
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

    available_inventory_columns = [
        column for column in inventory_columns
        if column in filtered_df.columns
    ]

    if not filtered_df.empty:
        st.dataframe(
            filtered_df[available_inventory_columns].round(2),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Download Inventory Report",
            filtered_df[available_inventory_columns].to_csv(index=False),
            file_name="retailmind_inventory_report.csv",
            mime="text/csv"
        )
    else:
        st.warning("No inventory records match the selected filters.")

    st.divider()
    st.subheader("Reorder Recommendations")

    reorder_df = filtered_df[
        filtered_df["Inventory Status"] == "Reorder Required"
    ].sort_values("Reorder Quantity", ascending=False)

    if not reorder_df.empty:
        reorder_chart = (
            reorder_df.groupby("Product ID", as_index=False)["Reorder Quantity"]
            .sum()
            .sort_values("Reorder Quantity", ascending=False)
            .head(10)
        )

        make_bar_chart(
            reorder_chart,
            "Product ID",
            "Reorder Quantity",
            AMBER
        )
    else:
        st.success("No products require reordering in this selection.")


elif page == "Forecasting":
    st.subheader("📈 Demand Forecasting")
    st.caption("Explore the model's demand estimates for available products.")

    if filtered_df.empty:
        st.warning("No forecasting records match the selected filters.")
    else:
        products = sorted(
            filtered_df["Product ID"].dropna().astype(str).unique().tolist()
        )

        if not products:
            st.info("No products are available for forecasting.")
        else:
            selected_product = st.selectbox("Choose Product", products)

            product_data = filtered_df[
                filtered_df["Product ID"].astype(str) == selected_product
            ]

            predicted = pd.to_numeric(
                product_data["Predicted Demand"],
                errors="coerce"
            ).mean()

            average_demand = pd.to_numeric(
                product_data["Average Daily Demand"],
                errors="coerce"
            ).mean()

            c1, c2 = st.columns(2)

            c1.metric(
                "Predicted Demand",
                f"{predicted:,.2f}" if pd.notna(predicted) else "N/A"
            )

            c2.metric(
                "Average Daily Demand",
                f"{average_demand:,.2f}" if pd.notna(average_demand) else "N/A"
            )

            st.divider()
            st.subheader("Predicted Demand by Product")

            prediction_data = (
                filtered_df.groupby("Product ID", as_index=False)[
                    "Predicted Demand"
                ]
                .mean()
                .dropna()
                .sort_values("Predicted Demand", ascending=False)
                .head(15)
            )

            if not prediction_data.empty:
                make_bar_chart(
                    prediction_data,
                    "Product ID",
                    "Predicted Demand",
                    BLUE
                )
            else:
                st.info("Predicted demand values are not available.")

    st.divider()
    st.subheader("Model Performance")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", "68.93")
    c2.metric("RMSE", "88.17")
    c3.metric("R² Score", "0.3333")

    st.info(
        "Random Forest performance on the held-out test dataset. "
        "These are experimental results based on demonstration data. "
        "The dashboard displays precomputed estimates, not live model inference."
    )


elif page == "Clustering":
    st.subheader("🔍 Product Clustering")
    st.caption("Explore similarities between store-product combinations.")

    if cluster_df.empty:
        st.warning("The clustering results file contains no records.")
    else:
        algorithm = st.selectbox(
            "Select Clustering Algorithm",
            ["Hierarchical Clustering", "DBSCAN"]
        )

        cluster_column = (
            "Hierarchical Cluster"
            if algorithm == "Hierarchical Clustering"
            else "DBSCAN Cluster"
        )

        if cluster_column not in cluster_df.columns:
            st.error(f"The clustering report does not contain '{cluster_column}'.")
        else:
            cluster_view = cluster_df.copy()

            if selected_store != "All Stores" and "Store ID" in cluster_view.columns:
                cluster_view = cluster_view[
                    cluster_view["Store ID"].astype(str) == selected_store
                ]

            if (
                selected_category != "All Categories"
                and "Category" in cluster_view.columns
            ):
                cluster_view = cluster_view[
                    cluster_view["Category"].astype(str) == selected_category
                ]

            if not cluster_view.empty:
                st.subheader("Cluster Distribution")

                cluster_counts = (
                    cluster_view[cluster_column]
                    .astype(str)
                    .value_counts()
                    .rename_axis("Cluster")
                    .reset_index(name="Count")
                )

                make_bar_chart(
                    cluster_counts,
                    "Cluster",
                    "Count",
                    TEAL
                )

                st.subheader("Cluster Details")

                detail_columns = [
                    column for column in
                    ["Store ID", "Product ID", cluster_column]
                    if column in cluster_view.columns
                ]

                st.dataframe(
                    cluster_view[detail_columns],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No clustering records match the selected filters.")


elif page == "Association Rules":
    st.subheader("🛒 Association Rule Mining")
    st.caption("Discover products frequently purchased together using Apriori.")

    required_rule_columns = [
        "Antecedents",
        "Consequents",
        "support",
        "confidence",
        "lift"
    ]

    missing_rule_columns = [
        column for column in required_rule_columns
        if column not in association_df.columns
    ]

    if missing_rule_columns:
        st.error(
            "The association rules report is missing: "
            + ", ".join(missing_rule_columns)
        )
    elif association_df.empty:
        st.warning("The association rules report contains no records.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05
            )

        with c2:
            lift_max = max(
                2.0,
                float(pd.to_numeric(association_df["lift"], errors="coerce").max())
            )

            min_lift = st.slider(
                "Minimum Lift",
                min_value=0.0,
                max_value=lift_max,
                value=1.0,
                step=0.1
            )

        rules = association_df[
            (pd.to_numeric(association_df["confidence"], errors="coerce") >= min_confidence)
            & (pd.to_numeric(association_df["lift"], errors="coerce") >= min_lift)
        ].sort_values("lift", ascending=False)

        st.metric("Matching Association Rules", f"{len(rules):,}")
        st.divider()
        st.subheader("Association Rules")

        if not rules.empty:
            st.dataframe(
                rules[required_rule_columns].head(30).round(3),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇ Download Association Rules",
                rules.to_csv(index=False),
                file_name="retailmind_association_rules.csv",
                mime="text/csv"
            )
        else:
            st.info("No rules match the selected confidence and lift thresholds.")


st.divider()
st.caption(
    "RetailMind | Machine Learning-Based Demand Forecasting "
    "and Inventory Intelligence System"
)
