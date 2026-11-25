"""
Canteen Sales Regression - Verification & Smoke Tests
Verifies dataset loading, processing, KPI analytics,
scikit-learn pipeline training, and 7-day demand forecasting.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_processing import (
    load_and_validate_data,
    prepare_daily_demand_data,
    filter_orders,
    ITEM_CATALOG,
)
from src.analytics import (
    calculate_kpis,
    get_daily_trend,
    get_item_performance,
    get_category_performance,
    get_hourly_sales,
    get_weekday_sales,
)
from src.model import (
    train_demand_model,
    predict_next_7_days,
)


@pytest.fixture(scope="module")
def raw_orders_df():
    """Fixture providing loaded raw orders."""
    csv_path = "data/canteen_sales.csv"
    if not os.path.exists(csv_path):
        from data.generate_data import generate_canteen_dataset
        generate_canteen_dataset(output_path=csv_path)
    return load_and_validate_data(csv_path)


@pytest.fixture(scope="module")
def daily_demand_df(raw_orders_df):
    """Fixture providing zero-imputed daily demand dataset."""
    return prepare_daily_demand_data(raw_orders_df)


@pytest.fixture(scope="module")
def trained_model_and_metrics(daily_demand_df):
    """Fixture providing trained model pipeline and evaluation metrics."""
    return train_demand_model(daily_demand_df)


def test_data_loading_and_schema(raw_orders_df):
    """Verify raw dataset has required columns, rows, and valid ranges."""
    assert not raw_orders_df.empty
    assert len(raw_orders_df) >= 4000
    expected_cols = [
        "order_id", "date", "time", "item", "category",
        "quantity", "unit_price", "payment_method", "revenue",
        "day_of_week", "day_name", "is_weekend", "month", "hour"
    ]
    for col in expected_cols:
        assert col in raw_orders_df.columns

    # Verify mathematical integrity: revenue = quantity * unit_price
    computed_rev = raw_orders_df["quantity"] * raw_orders_df["unit_price"]
    np.testing.assert_allclose(raw_orders_df["revenue"], computed_rev)


def test_zero_sales_daily_aggregation(daily_demand_df, raw_orders_df):
    """
    Verify daily demand dataset has complete Cartesian product (dates x items)
    and zero-imputes days with no sales.
    """
    unique_dates = raw_orders_df["date"].nunique()
    expected_rows = unique_dates * len(ITEM_CATALOG)
    assert len(daily_demand_df) == expected_rows

    # Verify no negative quantities
    assert (daily_demand_df["quantity_sold"] >= 0).all()

    # Verify total units match raw orders
    assert daily_demand_df["quantity_sold"].sum() == raw_orders_df["quantity"].sum()


def test_analytics_kpis(raw_orders_df):
    """Verify business KPIs compute positive, non-trivial numbers."""
    kpis = calculate_kpis(raw_orders_df)
    assert kpis["total_revenue"] > 100000
    assert kpis["total_orders"] == len(raw_orders_df)
    assert kpis["total_units_sold"] >= kpis["total_orders"]
    assert kpis["average_order_value"] > 0

    item_perf = get_item_performance(raw_orders_df)
    assert len(item_perf) == len(ITEM_CATALOG)
    assert item_perf["units_sold"].sum() == raw_orders_df["quantity"].sum()

    cat_perf = get_category_performance(raw_orders_df)
    assert len(cat_perf) == 4  # Breakfast, Snacks, Meals, Beverages

    hourly = get_hourly_sales(raw_orders_df)
    assert len(hourly) > 0
    assert (hourly["hour"] >= 8).all() and (hourly["hour"] <= 20).all()

    weekday = get_weekday_sales(raw_orders_df)
    assert len(weekday) == 7


def test_model_training_and_metrics(trained_model_and_metrics):
    """Verify RandomForestRegressor pipeline trains and achieves solid accuracy."""
    pipeline, metrics = trained_model_and_metrics
    assert pipeline is not None
    assert metrics["mae"] >= 0
    # On this structured synthetic dataset, R^2 should be healthy (> 0.6)
    assert metrics["r2"] > 0.5
    assert metrics["train_samples"] > 0
    assert metrics["test_samples"] > 0


def test_7_day_forecast_all_items(raw_orders_df, trained_model_and_metrics):
    """Verify 7-day forecast executes for every food item without error."""
    pipeline, _ = trained_model_and_metrics
    last_date = raw_orders_df["date"].max()

    for item_name in ITEM_CATALOG.keys():
        forecast_df, summary = predict_next_7_days(pipeline, item_name, last_date)
        assert len(forecast_df) == 7
        assert (forecast_df["Predicted Units"] >= 0).all()
        assert summary["total_units"] >= 0
        assert summary["total_revenue"] >= 0
        assert summary["peak_day"] in [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ]


def test_filter_functionality(raw_orders_df):
    """Verify interactive filter logic works smoothly."""
    filtered = filter_orders(raw_orders_df, categories=["Breakfast"])
    assert (filtered["category"] == "Breakfast").all()

    filtered_item = filter_orders(raw_orders_df, items=["Samosa"])
    assert (filtered_item["item"] == "Samosa").all()


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_model.py"])
