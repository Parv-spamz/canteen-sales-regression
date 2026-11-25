"""
Canteen Sales Regression - Data Processing Module
Handles CSV ingestion, data validation, temporal feature engineering,
and daily item-demand aggregation with zero-sales handling.
"""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Standard catalog definition for reference and zero-sales imputation
ITEM_CATALOG: Dict[str, Dict[str, any]] = {
    "Masala Dosa": {"category": "Breakfast", "unit_price": 60},
    "Veg Sandwich": {"category": "Snacks", "unit_price": 50},
    "Paneer Roll": {"category": "Snacks", "unit_price": 80},
    "Veg Thali": {"category": "Meals", "unit_price": 120},
    "Samosa": {"category": "Snacks", "unit_price": 25},
    "Cold Coffee": {"category": "Beverages", "unit_price": 70},
    "Lemonade": {"category": "Beverages", "unit_price": 40},
}

REQUIRED_COLUMNS = [
    "order_id", "date", "time", "item", "category",
    "quantity", "unit_price", "payment_method", "revenue"
]


def load_and_validate_data(filepath: str = "data/canteen_sales.csv") -> pd.DataFrame:
    """
    Loads raw canteen sales data, validates schema, cleans fields,
    and extracts temporal features.

    Returns:
        pd.DataFrame: Cleaned order-level DataFrame.
    Raises:
        FileNotFoundError: If the CSV file is not found.
        ValueError: If required columns are missing or data is empty.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Sales dataset file not found at: {filepath}")

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError("The loaded canteen dataset is empty.")

    # Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    # Drop null rows if any exist
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    # Convert date and extract calendar features
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["day_name"] = df["date"].dt.day_name()
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()

    # Extract hour of day from time column safely
    # Supports 'HH:MM:SS' and 'HH:MM'
    def parse_hour(time_val):
        try:
            return int(str(time_val).split(":")[0])
        except Exception:
            return 12

    df["hour"] = df["time"].apply(parse_hour)

    # Ensure numeric columns have correct types
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1).astype(int)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0.0)
    df["revenue"] = df["quantity"] * df["unit_price"]

    # Filter out invalid entries
    df = df[df["quantity"] > 0].copy()

    return df


def prepare_daily_demand_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms transactional order records into a balanced daily item-demand dataset.
    
    CRITICAL REQUIREMENT:
    Includes all (date x item) pairs. If an item had 0 sales on a particular date,
    quantity_sold is recorded as 0, avoiding survival bias in demand forecasting.

    Returns:
        pd.DataFrame: Aggregated daily demand table with features ready for ML.
    """
    if df.empty:
        raise ValueError("Cannot prepare daily demand from an empty DataFrame.")

    # Get all unique dates in the dataset sorted chronologically
    all_dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    all_items = list(ITEM_CATALOG.keys())

    # Build Cartesian product grid of (all_dates x all_items)
    grid = []
    for d in all_dates:
        for it in all_items:
            grid.append({
                "date": d,
                "item": it,
                "category": ITEM_CATALOG[it]["category"],
                "unit_price": ITEM_CATALOG[it]["unit_price"],
                "day_of_week": d.dayofweek,
                "day_name": d.day_name(),
                "is_weekend": int(d.dayofweek >= 5),
                "month": d.month,
                "month_name": d.month_name(),
            })
    grid_df = pd.DataFrame(grid)

    # Aggregate actual sales from transactions
    actual_sales = (
        df.groupby(["date", "item"], as_index=False)
        .agg(quantity_sold=("quantity", "sum"), revenue=("revenue", "sum"))
    )

    # Merge grid with actual sales
    daily_df = pd.merge(grid_df, actual_sales, on=["date", "item"], how="left")

    # Zero-fill days with no sales for that item
    daily_df["quantity_sold"] = daily_df["quantity_sold"].fillna(0).astype(int)
    daily_df["revenue"] = daily_df["revenue"].fillna(0.0)

    # Sort chronologically and by item
    daily_df = daily_df.sort_values(by=["date", "item"]).reset_index(drop=True)

    return daily_df


def filter_orders(
    df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    categories: Optional[List[str]] = None,
    items: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Applies interactive dashboard filters to order-level data.
    """
    filtered = df.copy()

    if start_date is not None:
        filtered = filtered[filtered["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        filtered = filtered[filtered["date"] <= pd.to_datetime(end_date)]
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    if items:
        filtered = filtered[filtered["item"].isin(items)]

    return filtered
