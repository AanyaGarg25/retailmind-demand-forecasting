from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException

from .schemas import ForecastRequest

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

app = FastAPI(
    title="RetailMind API",
    description="Demand Forecasting and Inventory Intelligence API",
    version="1.0"
)

inventory_df = pd.read_csv(
    REPORTS_DIR / "inventory_recommendations.csv",
    dtype={"Store ID": str, "Product ID": str},
    parse_dates=["Date"]
)

forecast_df = pd.read_csv(
    REPORTS_DIR / "demand_predictions.csv",
    dtype={"Store ID": str, "Product ID": str},
    parse_dates=["Date"]
)

retail_df = inventory_df.merge(
    forecast_df[
        ["Date", "Store ID", "Product ID", "Predicted Demand"]
    ],
    on=["Date", "Store ID", "Product ID"],
    how="left"
)

association_df = pd.read_csv(
    REPORTS_DIR / "association_rules.csv"
)

@app.get("/")
def home():
    return {
        "message": "Welcome to RetailMind API",
        "modules": [
            "Demand Forecasting",
            "Inventory Intelligence",
            "Association Rule Mining"
        ]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/forecast")
def forecast(request: ForecastRequest):

    result = retail_df[
        (retail_df["Store ID"] == request.store) &
        (retail_df["Product ID"] == request.item)
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail="Store-Product combination not found"
        )

    row = result.iloc[0]

    return {
        "date": row["Date"].strftime("%Y-%m-%d"),
        "store": row["Store ID"],
        "product": row["Product ID"],
        "predicted_demand": round(float(row["Predicted Demand"]), 2),
        "inventory_level": int(row["Inventory Level"]),
        "average_daily_demand": round(float(row["Average Daily Demand"]), 2),
        "stock_coverage": round(float(row["Stock Coverage"]), 2),
        "reorder_point": round(float(row["Reorder Point"]), 2),
        "reorder_quantity": int(row["Reorder Quantity"]),
        "inventory_status": row["Inventory Status"]
    }

@app.get("/inventory")
def inventory():
    return retail_df.fillna("").to_dict(orient="records")

@app.get("/association-rules")
def association_rules():
    return association_df[
        [
            "Antecedents",
            "Consequents",
            "support",
            "confidence",
            "lift"
        ]
    ].head(20).fillna("").to_dict(orient="records")