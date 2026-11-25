# Canteen Sales Regression
### Historical Sales Analytics & Machine Learning Demand Prediction for College Food Services

An end-to-end Python analytics and Machine Learning application designed for college canteens. It analyzes historical sales records, visualizes peak hours and category revenues, and uses a scikit-learn `RandomForestRegressor` pipeline to forecast daily food item preparation targets over the next seven days.

---

## 🚀 Key Features

1. **Executive KPI Dashboard:** Real-time metrics for Total Revenue, Total Orders, Units Sold, and Average Order Value (AOV).
2. **Rush Hour & Temporal Analytics:** Interactive Plotly charts identifying morning breakfast rush, lunchtime surge, evening tea breaks, and weekday vs. weekend patterns.
3. **Product & Category Insights:** Volume vs. revenue comparisons, department contribution shares, and payment method distributions.
4. **Machine Learning 7-Day Demand Forecast:** Select any food item to generate a 7-day preparation schedule with estimated units, gross revenue, and peak prep days.
5. **Data Leakage-Free ML Pipeline:** Scikit-learn `Pipeline` utilizing `OneHotEncoder`, `ColumnTransformer`, zero-sales demand imputation, and an 80/20 chronological train/test split.
6. **Transparent Academic Architecture:** Full dataset inspection, feature importance breakdown, and downloadable sample CSV.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Main programming language |
| **Pandas** | Data processing, feature extraction, and zero-demand aggregation |
| **NumPy** | Numerical computations and reproducible sampling |
| **Scikit-learn** | Supervised Machine Learning (`Pipeline`, `RandomForestRegressor`) |
| **Streamlit** | Modern, interactive web dashboard UI |
| **Plotly** | High-performance interactive charts and donut visualizations |
| **CSV** | Lightweight, zero-setup tabular storage |

---

## 📁 Project Structure

```
canteen-sales-regression/
│
├── app.py                     # Main Streamlit web application & UI
├── requirements.txt           # Python dependencies
├── README.md                  # Complete project documentation
├── canteen-sales-regression.pptx # Showcase presentation slides
├── PRESENTATION_NOTES.md      # Showcase presentation script & viva answers
├── PRESENTATION_SCRIPT.md     # 10-minute short presentation script
│
├── data/
│   ├── canteen_sales.csv      # Generated historical sales records (~6,000 orders)
│   └── generate_data.py       # Deterministic, reproducible dataset generator
│
├── src/
│   ├── data_processing.py     # Ingestion, validation, and zero-sales grid aggregation
│   ├── analytics.py           # Business KPIs, rush hour stats, product rankings
│   └── model.py               # Scikit-learn Pipeline & 7-day demand forecasting
│
└── tests/
    └── test_model.py          # Pytest verification and smoke tests
```

---

## 📦 Installation & Setup (Windows / macOS / Linux)

### 1. Clone or Open the Project Directory
Ensure your terminal is located in the project root:
```powershell
cd "c:\Users\aropa\OneDrive\Desktop\ML\New folder"
```

### 2. (Optional) Create a Python Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🏃 Running the Application

Launch the interactive dashboard with either of these commands:

### Option A: Standard Python Module Command (Recommended on Windows)
```powershell
python -m streamlit run app.py
```

### Option B: Quick Launcher Script
- In PowerShell:
  ```powershell
  .\run.ps1
  ```
- Or double-click `run.bat` in Windows File Explorer!

### Option C: If Streamlit is in your System PATH
```powershell
streamlit run app.py
```

The application will automatically open in your default browser at:
`http://localhost:8501`

---

## 🧪 Running Verification Tests

Run the complete automated test suite (schema validation, zero-sales aggregation, KPI math, ML pipeline training, and 7-day forecast verification):

```powershell
python -m pytest tests/test_model.py -v
```

All 6 test suites will run and pass within ~5 seconds.

---

## 📊 Dataset Catalog & Specifications

The dataset represents 180 days of canteen sales across 7 standard campus offerings:

| Food Item | Category | Unit Price | Typical Demand Profile |
|---|---|---|---|
| **Masala Dosa** | Breakfast | ₹60 | High morning peak (8–11 AM), moderate dinner |
| **Veg Sandwich** | Snacks | ₹50 | Steady demand throughout day (10 AM–6 PM) |
| **Paneer Roll** | Snacks | ₹80 | Lunch and evening snack favorite |
| **Veg Thali** | Meals | ₹120 | Heavy lunch surge (12–2:30 PM), high revenue |
| **Samosa** | Snacks | ₹25 | Highest volume item, peak evening tea rush |
| **Cold Coffee** | Beverages | ₹70 | Midday and afternoon study refreshment |
| **Lemonade** | Beverages | ₹40 | Midday hydration refresher |

> **Academic Transparency Note:** The historical dataset (`data/canteen_sales.csv`) is synthetically generated using realistic academic scheduling parameters (`random_state=42`) for reproducibility and educational demonstration. It can be regenerated at any time using `python data/generate_data.py`.

---

## 🧠 Machine Learning Model Details

- **Algorithm:** `RandomForestRegressor` (100 estimators, max depth 12, random state 42).
- **Target Variable:** `quantity_sold` (daily total units of a food item).
- **Features Used:**
  - `item` (Categorical, OneHotEncoded)
  - `category` (Categorical, OneHotEncoded)
  - `day_name` (Categorical, OneHotEncoded)
  - `unit_price` (Numerical)
  - `is_weekend` (Binary 0/1)
  - `month` (Numerical)
  - `day_of_week` (Numerical 0–6)
- **Train / Test Split:** 80% chronological training (Days 1–144: 1,008 samples), 20% holdout testing (Days 145–180: 252 samples).
- **Test Performance:**
  - **Mean Absolute Error (MAE):** ~1.53 units (predictions are off by only ~1.5 portions on average).
  - **R² Score:** ~0.716 (explains 71.6% of daily demand variance).
- **Kitchen Output:** Non-negative integer predictions with expected gross revenue calculations for the upcoming 7 days.

---

## 🎓 Showcase Presentation Guide

For comprehensive viva preparation, explanation of MAE/R², anticipated evaluator questions, and a step-by-step presentation script, refer to:
👉 [PRESENTATION_NOTES.md](PRESENTATION_NOTES.md)
