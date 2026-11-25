"""
Canteen Sales Regression - Machine Learning Engine
Implements Random Forest Regression with a scikit-learn Pipeline
for daily food item demand prediction and 7-day kitchen forecasting.
"""

from typing import Dict, Any, Tuple
from datetime import timedelta
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from src.data_processing import ITEM_CATALOG

# Define feature columns
CATEGORICAL_FEATURES = ["item", "category", "day_name"]
NUMERICAL_FEATURES = ["unit_price", "is_weekend", "month", "day_of_week"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COLUMN = "quantity_sold"


def build_pipeline() -> Pipeline:
    """
    Constructs a robust scikit-learn Pipeline combining
    OneHotEncoder preprocessing with RandomForestRegressor.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                "passthrough",
                NUMERICAL_FEATURES,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=12,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    return pipeline


def train_demand_model(
    daily_df: pd.DataFrame,
    test_size: float = 0.20,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains the demand forecasting model using an 80/20 chronological train/test split.
    Chronological splitting prevents lookahead bias in time-series demand estimation.

    Returns:
        pipeline: Trained scikit-learn Pipeline.
        metrics: Dictionary containing MAE, R², dataset counts, and feature info.
    """
    if daily_df.empty or len(daily_df) < 50:
        raise ValueError("Insufficient daily demand data to train the model.")

    # Sort strictly chronologically
    sorted_df = daily_df.sort_values("date").reset_index(drop=True)

    # Determine date-based split point (80% train dates, 20% test dates)
    unique_dates = sorted_df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    num_train_dates = int(len(unique_dates) * (1.0 - test_size))
    split_date = unique_dates.iloc[num_train_dates]

    train_df = sorted_df[sorted_df["date"] < split_date].copy()
    test_df = sorted_df[sorted_df["date"] >= split_date].copy()

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET_COLUMN]

    # Train model pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Predictions on test set
    y_pred_test = pipeline.predict(X_test)
    # Clip negative predictions to 0
    y_pred_test = np.clip(y_pred_test, 0, None)

    # Also evaluate on train set for diagnostic comparison
    y_pred_train = pipeline.predict(X_train)
    y_pred_train = np.clip(y_pred_train, 0, None)

    mae_test = float(mean_absolute_error(y_test, y_pred_test))
    r2_test = float(r2_score(y_test, y_pred_test))
    mae_train = float(mean_absolute_error(y_train, y_pred_train))
    r2_train = float(r2_score(y_train, y_pred_train))

    # Extract feature names & importances from RandomForest
    try:
        ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        feature_names = cat_feature_names + NUMERICAL_FEATURES
        rf_regressor = pipeline.named_steps["regressor"]
        importances = rf_regressor.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).head(8)
    except Exception:
        importance_df = pd.DataFrame()

    metrics = {
        "model_name": "Random Forest Regressor (Pipeline)",
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "total_samples": len(sorted_df),
        "train_date_range": f"{train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')}",
        "test_date_range": f"{test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')}",
        "mae": round(mae_test, 2),
        "r2": round(r2_test, 3),
        "train_mae": round(mae_train, 2),
        "train_r2": round(r2_train, 3),
        "top_features": importance_df.to_dict(orient="records") if not importance_df.empty else [],
    }

    return pipeline, metrics


def predict_next_7_days(
    pipeline: Pipeline,
    selected_item: str,
    last_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates a 7-day demand forecast for a chosen food item starting
    the day immediately after `last_date`.

    Returns:
        forecast_df: Table with Date, Day, Predicted Units, Expected Revenue.
        summary: Aggregates including total units, expected revenue, peak day.
    """
    if selected_item not in ITEM_CATALOG:
        raise ValueError(f"Unknown food item: '{selected_item}'")

    item_meta = ITEM_CATALOG[selected_item]
    unit_price = item_meta["unit_price"]
    category = item_meta["category"]

    # Construct the future 7 calendar days
    future_rows = []
    for day_offset in range(1, 8):
        future_dt = last_date + timedelta(days=day_offset)
        future_rows.append({
            "date": future_dt,
            "date_str": future_dt.strftime("%Y-%m-%d"),
            "item": selected_item,
            "category": category,
            "unit_price": unit_price,
            "day_of_week": future_dt.dayofweek,
            "day_name": future_dt.day_name(),
            "is_weekend": int(future_dt.dayofweek >= 5),
            "month": future_dt.month,
        })

    future_df = pd.DataFrame(future_rows)
    X_future = future_df[ALL_FEATURES]

    # Predict using trained pipeline
    raw_preds = pipeline.predict(X_future)

    # Process predictions: integer units, never negative
    predicted_units = [max(0, int(round(p))) for p in raw_preds]
    expected_revenues = [units * unit_price for units in predicted_units]

    forecast_df = pd.DataFrame({
        "Date": future_df["date_str"],
        "Day": future_df["day_name"],
        "Is Weekend": future_df["is_weekend"].apply(lambda x: "Yes" if x == 1 else "No"),
        "Predicted Units": predicted_units,
        "Unit Price (₹)": unit_price,
        "Expected Revenue (₹)": expected_revenues,
    })

    # Find peak day
    max_idx = forecast_df["Predicted Units"].idxmax()
    peak_row = forecast_df.iloc[max_idx]

    summary = {
        "item": selected_item,
        "category": category,
        "unit_price": unit_price,
        "total_units": int(forecast_df["Predicted Units"].sum()),
        "total_revenue": float(forecast_df["Expected Revenue (₹)"].sum()),
        "peak_day": peak_row["Day"],
        "peak_date": peak_row["Date"],
        "peak_units": int(peak_row["Predicted Units"]),
        "avg_daily_predicted": round(float(forecast_df["Predicted Units"].mean()), 1),
    }

    return forecast_df, summary
