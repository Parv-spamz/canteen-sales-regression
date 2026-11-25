# Canteen Sales Regression

Sales analytics and 7-day demand forecasting for a college canteen.

The project takes an order log, turns it into a per-item daily demand table, and trains a Random Forest to predict how many portions of each item the kitchen should prepare for the coming week. A Streamlit dashboard sits on top for exploring the sales history and reading the forecast.

**Authors:** Parv Arora (2401420008) and Anshika Trikha (2401420044)

## Results

Evaluated on the last 36 days, which the model never sees during training.

| Model | R² | MAE (units) |
|---|---|---|
| Predict the overall mean | 0.00 | 2.67 |
| Predict the per-item mean | 0.30 | 2.35 |
| Linear regression | 0.62 | 1.79 |
| **Random Forest (used)** | **0.716** | **1.53** |

On average the forecast is off by about one and a half portions per item per day. Training R² is 0.81, so the gap to the test score is small.

The three baseline rows are not produced by the app; they were computed separately on the same split to give the Random Forest figure some context.

## How the model works

**Data.** Real canteen billing data is not public, so `data/generate_data.py` produces a synthetic log with a fixed seed (42). It covers 180 days (1 May to 27 Oct 2024), 7 menu items and about 6,000 orders, with meal-time rushes, a weekend drop and semester seasonality built in. Running the script again gives the same file.

**Demand table.** The log has one row per order, but the model needs one row per item per day. Orders are aggregated to 180 days × 7 items = 1,260 rows, and days where an item did not sell are filled with 0. Skipping that step would show the model only days with sales and push its predictions upward.

**Features.**

| Feature | Type |
|---|---|
| `item`, `category`, `day_name` | categorical, one-hot encoded |
| `unit_price`, `month`, `day_of_week` | numeric |
| `is_weekend` | binary |

The target is `quantity_sold`, the units of an item sold in a day.

**Split.** Chronological, not random: the first 144 days (1,008 rows) train the model and the last 36 days (252 rows) test it. A random shuffle would let future days leak into training.

**Pipeline.** A single scikit-learn `Pipeline`: a `ColumnTransformer` for the encoding, followed by a `RandomForestRegressor` with 100 trees, `max_depth=12`, `min_samples_split=4`, `min_samples_leaf=2` and `random_state=42`. Predictions are clipped at zero and rounded to whole portions.

## Dashboard

| Tab | Contents |
|---|---|
| Overview | Revenue, orders, units sold and average order value; revenue trend with a 7-day moving average; sales by hour and by weekday; revenue by category |
| Product Insights | Volume against revenue per item, category shares, payment methods |
| Demand Prediction | Pick an item to get a 7-day preparation schedule with units, expected revenue and the peak day |
| Dataset and Model Details | Raw records, feature table, feature importances, sample CSV download |

The sidebar filters by date range, category and item, and every chart updates with it.

## Getting started

Requires Python 3.10 or newer.

```powershell
git clone https://github.com/Parv-spamz/canteen-sales-regression.git
cd canteen-sales-regression
pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`. On Windows, `run.ps1` or `run.bat` does the same thing. To rebuild the dataset, run `python data/generate_data.py`.

## Project layout

```
app.py                  Streamlit app
requirements.txt
data/
  canteen_sales.csv     generated order log
  generate_data.py      seeded dataset generator
src/
  data_processing.py    loading, validation, demand-table aggregation
  analytics.py          KPIs, rush-hour stats, product rankings
  model.py              pipeline, training, 7-day forecast
```

## Menu in the dataset

| Item | Category | Price | Demand pattern |
|---|---|---|---|
| Masala Dosa | Breakfast | ₹60 | Morning peak, smaller dinner bump |
| Veg Sandwich | Snacks | ₹50 | Steady from 10 AM to 6 PM |
| Paneer Roll | Snacks | ₹80 | Lunch and evening |
| Veg Thali | Meals | ₹120 | Heavy lunch rush, highest revenue |
| Samosa | Snacks | ₹25 | Highest volume, evening tea rush |
| Cold Coffee | Beverages | ₹70 | Midday and afternoon |
| Lemonade | Beverages | ₹40 | Midday |

## Limitations

- The data is synthetic, so the scores say how well the pipeline recovers the patterns we generated, not how it would do on a real canteen.
- Features are calendar and item information only. Weather, exams, events and holidays are not modelled.
- One fixed chronological split, no cross-validation or hyperparameter search.

## Stack

Python, pandas, NumPy, scikit-learn, Streamlit, Plotly.
