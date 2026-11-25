"""
Canteen Sales Regression - Synthetic Data Generator
Generates realistic historical canteen sales data (180 days, ~5,000 orders)
with natural weekday/weekend patterns, meal-time peak hours, and product dynamics.

NOTE: This dataset is synthetic, designed for academic demonstration and machine learning training.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Set deterministic seed for complete reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Product catalog specifications
ITEMS = [
    {"item": "Masala Dosa", "category": "Breakfast", "unit_price": 60, "base_orders": 6},
    {"item": "Veg Sandwich", "category": "Snacks", "unit_price": 50, "base_orders": 5},
    {"item": "Paneer Roll", "category": "Snacks", "unit_price": 80, "base_orders": 4},
    {"item": "Veg Thali", "category": "Meals", "unit_price": 120, "base_orders": 6},
    {"item": "Samosa", "category": "Snacks", "unit_price": 25, "base_orders": 8},
    {"item": "Cold Coffee", "category": "Beverages", "unit_price": 70, "base_orders": 5},
    {"item": "Lemonade", "category": "Beverages", "unit_price": 40, "base_orders": 4},
]

ITEM_DICT = {it["item"]: it for it in ITEMS}

# Payment methods and probabilities
PAYMENT_METHODS = ["UPI", "Cash", "Card"]
PAYMENT_PROBS = [0.65, 0.25, 0.10]

# Weekday demand multipliers (Monday=0 to Sunday=6)
DAY_MULTIPLIERS = {
    0: 1.05,  # Monday: High campus return
    1: 1.00,  # Tuesday: Steady
    2: 1.02,  # Wednesday: Mid-week peak
    3: 1.05,  # Thursday: Busy lab day
    4: 1.15,  # Friday: Peak campus social & snack rush
    5: 0.50,  # Saturday: Partial labs / weekend students
    6: 0.28,  # Sunday: Hostellers only
}

# Academic calendar monthly seasonality (May to October)
MONTH_MULTIPLIERS = {
    5: 0.88,  # May: Summer semester / exams
    6: 0.90,  # June: End of term
    7: 1.08,  # July: New semester inauguration rush
    8: 1.12,  # August: Full attendance
    9: 1.05,  # September: Sports & fest prep
    10: 1.00, # October: Midterms & steady demand
}


def sample_order_time(item_name: str) -> str:
    """
    Generates a realistic time of day based on the item's meal profile.
    """
    if item_name == "Masala Dosa":
        # Heavy morning peak (8-11 AM), small evening dinner (6-8 PM)
        if random.random() < 0.80:
            hour = random.choices([8, 9, 10, 11], weights=[3, 4, 3, 1])[0]
        else:
            hour = random.choices([18, 19, 20], weights=[2, 3, 2])[0]

    elif item_name == "Veg Thali":
        # Lunch rush (12-3 PM: 85%), Light dinner (7-8 PM: 15%)
        if random.random() < 0.85:
            hour = random.choices([12, 13, 14], weights=[4, 4, 2])[0]
        else:
            hour = random.choices([19, 20], weights=[3, 2])[0]

    elif item_name == "Samosa":
        # Afternoon tea rush (3-6 PM: 75%), mid-morning break (10-11 AM: 25%)
        if random.random() < 0.75:
            hour = random.choices([15, 16, 17, 18], weights=[3, 4, 3, 1])[0]
        else:
            hour = random.choices([10, 11], weights=[3, 2])[0]

    elif item_name == "Veg Sandwich":
        # Distributed snacks throughout daytime (9 AM - 6 PM)
        hour = random.choices(
            [9, 10, 11, 12, 14, 15, 16, 17, 18],
            weights=[2, 3, 2, 2, 2, 4, 4, 3, 1]
        )[0]

    elif item_name == "Paneer Roll":
        # Lunch to evening snacks (12 PM - 7 PM)
        hour = random.choices(
            [12, 13, 14, 15, 16, 17, 18, 19],
            weights=[3, 3, 2, 2, 3, 4, 3, 2]
        )[0]

    elif item_name == "Cold Coffee":
        # Warm afternoon & study sessions (11 AM - 6 PM)
        hour = random.choices(
            [11, 12, 13, 14, 15, 16, 17, 18],
            weights=[2, 2, 2, 3, 4, 4, 3, 1]
        )[0]

    elif item_name == "Lemonade":
        # Hot midday hours (10 AM - 4 PM)
        hour = random.choices(
            [10, 11, 12, 13, 14, 15, 16],
            weights=[2, 3, 4, 4, 3, 2, 1]
        )[0]

    else:
        hour = random.randint(8, 20)

    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def sample_order_quantity(item_name: str) -> int:
    """
    Returns realistic order quantity (1-4 units) weighted by item type.
    """
    if item_name in ["Samosa", "Cold Coffee", "Lemonade"]:
        # Often shared among friends
        return random.choices([1, 2, 3, 4], weights=[0.50, 0.38, 0.09, 0.03])[0]
    elif item_name == "Veg Thali":
        # Almost always single meal, rarely 2
        return random.choices([1, 2], weights=[0.88, 0.12])[0]
    else:
        # Masala Dosa, Veg Sandwich, Paneer Roll
        return random.choices([1, 2, 3], weights=[0.68, 0.27, 0.05])[0]


def generate_canteen_dataset(
    start_date: str = "2024-05-01",
    num_days: int = 180,
    output_path: str = "data/canteen_sales.csv"
) -> pd.DataFrame:
    """
    Generates approximately 5,000 realistic order records across num_days.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    records = []
    order_counter = 10001

    for day_offset in range(num_days):
        current_date = start_dt + timedelta(days=day_offset)
        weekday = current_date.weekday()
        month = current_date.month

        day_factor = DAY_MULTIPLIERS.get(weekday, 1.0)
        month_factor = MONTH_MULTIPLIERS.get(month, 1.0)

        # For each item, determine daily order count
        for item_info in ITEMS:
            it_name = item_info["item"]
            cat = item_info["category"]
            price = item_info["unit_price"]
            base = item_info["base_orders"]

            # Expected order count for this day
            expected_orders = base * day_factor * month_factor

            # Natural Gaussian noise (standard deviation = 0.65)
            actual_orders = max(0, int(round(np.random.normal(expected_orders, 0.65))))

            for _ in range(actual_orders):
                order_id = f"ORD-{order_counter}"
                order_counter += 1

                time_str = sample_order_time(it_name)
                quantity = sample_order_quantity(it_name)
                revenue = quantity * price
                payment = random.choices(PAYMENT_METHODS, weights=PAYMENT_PROBS, k=1)[0]

                records.append({
                    "order_id": order_id,
                    "date": current_date.strftime("%Y-%m-%d"),
                    "time": time_str,
                    "item": it_name,
                    "category": cat,
                    "quantity": quantity,
                    "unit_price": price,
                    "payment_method": payment,
                    "revenue": revenue
                })

    df = pd.DataFrame(records)

    # Sort chronologically by date and time
    df = df.sort_values(by=["date", "time"]).reset_index(drop=True)

    # Re-assign sequential order IDs in chronological order
    df["order_id"] = [f"ORD-{10001 + i}" for i in range(len(df))]

    # Ensure parent dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} orders across {num_days} days.")
    print(f"Saved to: {output_path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total revenue: INR {df['revenue'].sum():,.2f}")
    return df


if __name__ == "__main__":
    generate_canteen_dataset()
