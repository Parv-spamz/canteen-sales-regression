"""
Canteen Sales Regression - Analytics Engine
Computes business KPIs, category metrics, product rankings,
hourly rush patterns, and temporal revenue trends.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


def format_inr(amount: float, short: bool = False) -> str:
    """
    Formats currency in Indian Rupee format (e.g., ₹1,45,200 or ₹1.45L).
    """
    if np.isnan(amount) or amount == 0:
        return "₹0"
    if short:
        if amount >= 10000000:
            return f"₹{amount / 10000000:.2f} Cr"
        elif amount >= 100000:
            return f"₹{amount / 100000:.2f} L"
        elif amount >= 1000:
            return f"₹{amount / 1000:.1f}k"
    return f"₹{amount:,.2f}"


def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates primary executive KPIs from the order records.
    """
    if df.empty:
        return {
            "total_revenue": 0.0,
            "total_orders": 0,
            "total_units_sold": 0,
            "average_order_value": 0.0,
            "formatted_revenue": "₹0",
            "formatted_revenue_short": "₹0",
            "formatted_orders": "0",
            "formatted_units": "0",
            "formatted_aov": "₹0.00",
        }

    total_rev = float(df["revenue"].sum())
    total_orders = int(df["order_id"].nunique())
    total_units = int(df["quantity"].sum())
    aov = total_rev / total_orders if total_orders > 0 else 0.0

    return {
        "total_revenue": total_rev,
        "total_orders": total_orders,
        "total_units_sold": total_units,
        "average_order_value": aov,
        "formatted_revenue": f"₹{total_rev:,.2f}",
        "formatted_revenue_short": format_inr(total_rev, short=True),
        "formatted_orders": f"{total_orders:,}",
        "formatted_units": f"{total_units:,}",
        "formatted_aov": f"₹{aov:.2f}",
    }


def get_daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily revenue, volume, and rolling 7-day averages.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "revenue", "quantity", "order_count", "rolling_7d_revenue"])

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum"),
            order_count=("order_id", "nunique"),
        )
        .sort_values("date")
    )

    daily["rolling_7d_revenue"] = daily["revenue"].rolling(window=7, min_periods=1).mean()
    return daily


def get_item_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates performance by food item: units sold, revenue, share of sales.
    """
    if df.empty:
        return pd.DataFrame(columns=["item", "category", "unit_price", "units_sold", "revenue", "order_count", "revenue_share_pct"])

    total_revenue = df["revenue"].sum()

    item_perf = (
        df.groupby(["item", "category", "unit_price"], as_index=False)
        .agg(
            units_sold=("quantity", "sum"),
            revenue=("revenue", "sum"),
            order_count=("order_id", "nunique"),
        )
    )

    item_perf["revenue_share_pct"] = (
        (item_perf["revenue"] / total_revenue * 100) if total_revenue > 0 else 0.0
    ).round(1)

    # Calculate average daily units sold
    num_days = max(1, df["date"].nunique())
    item_perf["avg_daily_units"] = (item_perf["units_sold"] / num_days).round(1)

    return item_perf.sort_values(by="units_sold", ascending=False).reset_index(drop=True)


def get_category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates metrics by food category (Breakfast, Snacks, Meals, Beverages).
    """
    if df.empty:
        return pd.DataFrame(columns=["category", "units_sold", "revenue", "revenue_share_pct"])

    total_revenue = df["revenue"].sum()

    cat_perf = (
        df.groupby("category", as_index=False)
        .agg(
            units_sold=("quantity", "sum"),
            revenue=("revenue", "sum"),
            order_count=("order_id", "nunique"),
        )
    )

    cat_perf["revenue_share_pct"] = (
        (cat_perf["revenue"] / total_revenue * 100) if total_revenue > 0 else 0.0
    ).round(1)

    return cat_perf.sort_values(by="revenue", ascending=False).reset_index(drop=True)


def get_hourly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates sales by hour of the day (8:00 to 20:00).
    """
    if df.empty:
        return pd.DataFrame(columns=["hour", "time_label", "revenue", "quantity", "order_count"])

    hourly = (
        df.groupby("hour", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum"),
            order_count=("order_id", "nunique"),
        )
    )

    # Format human-friendly time labels (e.g., "8 AM", "12 PM")
    def format_hour(h: int) -> str:
        if h == 0:
            return "12 AM"
        elif h < 12:
            return f"{h} AM"
        elif h == 12:
            return "12 PM"
        else:
            return f"{h - 12} PM"

    hourly["time_label"] = hourly["hour"].apply(format_hour)
    hourly = hourly.sort_values("hour").reset_index(drop=True)
    return hourly


def get_weekday_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates sales by weekday in proper chronological order (Monday to Sunday).
    """
    if df.empty:
        return pd.DataFrame(columns=["day_name", "day_of_week", "revenue", "quantity", "order_count", "avg_daily_revenue"])

    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]

    day_counts = df.groupby(["day_name", "date"]).size().reset_index().groupby("day_name").size()

    grouped = (
        df.groupby(["day_name", "day_of_week"], as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum"),
            order_count=("order_id", "nunique"),
        )
    )

    grouped["distinct_days"] = grouped["day_name"].map(day_counts).fillna(1)
    grouped["avg_daily_revenue"] = (grouped["revenue"] / grouped["distinct_days"]).round(2)

    # Order by day_of_week
    grouped = grouped.sort_values("day_of_week").reset_index(drop=True)
    return grouped


def get_product_insights_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Provides top-level highlights and standout statistics for the Product Insights tab.
    """
    if df.empty:
        return {
            "best_selling_item": "N/A",
            "best_selling_units": 0,
            "highest_revenue_item": "N/A",
            "highest_revenue_amt": 0.0,
            "least_selling_item": "N/A",
            "least_selling_units": 0,
            "top_payment_method": "N/A",
        }

    item_perf = get_item_performance(df)
    best_seller = item_perf.iloc[0]
    highest_rev = item_perf.sort_values(by="revenue", ascending=False).iloc[0]
    least_seller = item_perf.iloc[-1]

    payment_breakdown = df["payment_method"].value_counts(normalize=True) * 100
    top_payment = payment_breakdown.index[0]
    top_payment_pct = payment_breakdown.iloc[0]

    return {
        "best_selling_item": best_seller["item"],
        "best_selling_units": int(best_seller["units_sold"]),
        "highest_revenue_item": highest_rev["item"],
        "highest_revenue_amt": float(highest_rev["revenue"]),
        "least_selling_item": least_seller["item"],
        "least_selling_units": int(least_seller["units_sold"]),
        "top_payment_method": f"{top_payment} ({top_payment_pct:.1f}%)",
    }
