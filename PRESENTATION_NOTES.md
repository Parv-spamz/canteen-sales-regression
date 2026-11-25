# Canteen Sales Regression – Presentation & Viva Guide
**Sales Analytics & Machine Learning Demand Prediction for College Food Services**

---

## 1. Project Introduction
*“Good morning / afternoon evaluators and teachers. Today we present **Canteen Sales Regression**, a data analytics and machine learning solution designed to solve one of the most persistent operational bottlenecks in university campuses: balancing kitchen preparation with volatile student demand.”*

College canteens face a daily dilemma:
- If they prepare **too much food**, excess items spoil, leading to severe financial loss and unhygienic food waste.
- If they prepare **too little food**, popular items sell out within the first 30 minutes of lunch or break time, resulting in frustrated students and lost revenue.

Our application bridges this gap by combining **historical sales analytics** with **supervised machine learning regression**, providing canteen staff with an actionable, 7-day food preparation roadmap.

---

## 2. The Real-World Problem
University food services operate in a uniquely volatile environment:
1. **Meal-Specific Peak Windows:** Unlike regular restaurants with smooth traffic, campus canteens experience extreme demand bursts matching class schedules (e.g., 08:00–10:30 AM breakfast rush, 12:00–02:30 PM lunch surge, and 04:00–06:00 PM tea/snack rush).
2. **Weekday vs. Weekend Drop-offs:** Student traffic plummets on weekends (Saturdays drop ~50%, Sundays drop ~70%), but chefs often fail to scale batch cooking down appropriately.
3. **Item-Specific Dynamics:** Samosas are bought in high frequency as impulse snacks, while Veg Thalis follow strict lunch-hour demand. Cooking Thalis in the evening or Dosa batter at night leads to wastage.
4. **Lack of Digital Planning:** Most college canteens rely purely on guesswork and chef memory, with zero forecasting infrastructure.

---

## 3. The Proposed Solution
Our solution provides a two-in-one system:
1. **Descriptive Analytics (Understanding the Past):**
   - Tracks total revenue, order volume, and average order value (AOV).
   - Visualizes sales by hour of the day and day of the week.
   - Identifies high-margin vs. high-velocity food items.
2. **Predictive Machine Learning (Planning the Future):**
   - Employs a scikit-learn `RandomForestRegressor` pipeline.
   - Accurately estimates how many units of any food item the kitchen should prepare each day over the next 7 days.
   - Converts predicted food units into expected gross revenue to assist procurement.

---

## 4. System Architecture
The application is built on a clean, decoupled 5-tier architecture:

```
┌────────────────────────────────────────────────────────┐
│ 1. Canteen Sales Dataset (CSV · 180 Days · ~6,000 Tx) │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Data Processing Layer (Pandas & NumPy)              │
│    - Schema validation & datetime parsing              │
│    - Cartesian grid aggregation (Dates × 7 Items)      │
│    - Zero-sales imputation for true demand tracking    │
└────────────┬─────────────────────────────┬─────────────┘
             │                             │
             ▼                             ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│ 3A. Analytics Engine      │ │ 3B. ML Engine             │
│  - Executive KPIs         │ │  - ColumnTransformer OHE  │
│  - Hourly rush analysis   │ │  - RandomForestRegressor  │
│  - Product rankings       │ │  - 80/20 Chrono Split     │
└────────────┬──────────────┘ └─────────────┬─────────────┘
             │                             │
             └──────────────┬──────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 4. Interactive Streamlit Dashboard                     │
│    - Executive KPI Cards & Dynamic Filters             │
│    - Interactive Plotly Visualizations                 │
│    - 7-Day Kitchen Batch Forecast & Revenue Projections│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 5. Practical Outcome: Zero Waste & Optimal Inventory   │
└────────────────────────────────────────────────────────┘
```

---

## 5. Technologies Used and Why

| Technology | Role | Why It Was Chosen |
|---|---|---|
| **Python** | Core Language | Industry-standard language for Data Science and Machine Learning with rich library ecosystem. |
| **Pandas** | Data Processing | Vectorized data manipulation, temporal grouping, and zero-imputation aggregation. |
| **NumPy** | Numerical Operations | Fast numerical operations, array indexing, and reproducible statistical noise simulation. |
| **Scikit-learn** | Machine Learning | Robust `Pipeline`, `ColumnTransformer`, `OneHotEncoder`, and `RandomForestRegressor`. |
| **Streamlit** | Dashboard Frontend | Allows rapid creation of responsive, presentation-ready web applications entirely in Python without frontend complexity. |
| **Plotly** | Interactive Charts | Interactive tooltips, zoom, pan, hover templates, and responsive layouts that surpass static Matplotlib charts. |
| **CSV** | Data Storage | Lightweight, portable, human-readable, and zero-configuration database substitute. |

---

## 6. Dataset Explanation
- **Synthetic Academic Nature:** College canteens rarely publish point-of-sale databases due to privacy or informal accounting. We generated a realistic synthetic dataset (`data/canteen_sales.csv`) with a fixed seed (`random_state=42`) for 100% reproducibility.
- **Span:** 180 consecutive calendar days (May 1, 2024 to October 27, 2024).
- **Records:** 5,970 individual customer orders.
- **Menu Catalog:**
  1. *Masala Dosa* (Breakfast, ₹60)
  2. *Veg Sandwich* (Snacks, ₹50)
  3. *Paneer Roll* (Snacks, ₹80)
  4. *Veg Thali* (Meals, ₹120)
  5. *Samosa* (Snacks, ₹25)
  6. *Cold Coffee* (Beverages, ₹70)
  7. *Lemonade* (Beverages, ₹40)
- **Key Columns:** `order_id`, `date`, `time`, `item`, `category`, `quantity`, `unit_price`, `payment_method`, `revenue`.
- **Realistic Built-in Patterns:**
  - Meal-time hourly peaks (Breakfast 8–11 AM, Lunch 12–2:30 PM, Snacks 4–6 PM).
  - Weekday surge vs. Weekend drop.
  - Semester seasonality (July-August high attendance vs. May exams).
  - Payment split: 65% UPI, 25% Cash, 10% Card.

---

## 7. Machine Learning Algorithm Explanation
### Why Random Forest Regressor?
1. **Handles Non-Linear Demand Patterns:** Canteen demand does not scale in a simple straight line (linear regression fails here). Random Forest captures sudden weekend drops and non-linear interactions effortlessly.
2. **Ensemble of Decision Trees:** It constructs 100 decorrelated decision trees, each trained on bootstrap subsets of the data. By averaging their outputs (*bagging*), it significantly reduces prediction variance and prevents overfitting.
3. **No Over-Engineering:** It avoids the complexity, opacity, and data hunger of neural networks (LSTMs, GRUs) while being vastly more robust than single decision trees.
4. **Feature Interpretability:** Random Forest allows us to inspect feature importance, confirming to evaluators that Day of Week and Weekend flags are the primary drivers of food demand.

---

## 8. Training, Testing, and Data Leakage Prevention
### 1. Zero-Sales Imputation
In raw order logs, days where zero units of an item are sold simply don't have a row. If an ML model is trained only on days with orders, it suffers from *survival bias* and systematically over-predicts demand. We create a complete **Cartesian Grid of 180 Dates × 7 Items = 1,260 daily records**, filling unsold items with `0`.

### 2. Chronological 80/20 Train/Test Split
In standard ML, data is often shuffled randomly (`train_test_split(shuffle=True)`). However, for demand forecasting, **shuffling causes severe temporal data leakage** (using tomorrow’s data to predict yesterday). 
Instead, we perform a **chronological split**:
- **Training Set (80%):** Days 1 to 144 (May 1 to September 21, 2024) — 1,008 samples.
- **Testing Holdout (20%):** Days 145 to 180 (September 22 to October 27, 2024) — 252 samples.

---

## 9. MAE and R² Explained in Beginner-Friendly Language

### Mean Absolute Error (MAE):
- **Definition:** The average difference between the model's predicted quantity and the actual quantity sold.
- **In Plain Words:** *“If the actual number of Samosas sold was 18, and our model predicted 16, the error is 2. The MAE calculates this error across all 252 test days.”*
- **Our Result:** **MAE ≈ 1.5 units**. On average, our kitchen forecast is off by less than 2 portions of food, which is well within acceptable kitchen safety margins!

### R² (R-Squared) Score:
- **Definition:** The proportion of variance in food demand that the model successfully explains using the input features.
- **In Plain Words:** *“A baseline model that simply guesses the average demand every single day would score R² = 0. Our model scores **R² ≈ 0.72**, meaning it explains 72% of the variations in daily student demand using day of the week, price, and item category!”*

---

## 10. Step-by-Step Demonstration Script for the Showcase

| Step | Action on Screen | What to Say to the Evaluator |
|---|---|---|
| **1. Hook & Overview** | Open Dashboard header and executive KPI cards. | *"Good morning. Here is Canteen Sales Regression. Over the last 6 months, our canteen logged ₹5.1 Lakhs in revenue across 5,970 orders, with an average ticket size of ₹85."* |
| **2. Interactive Filtering** | Toggle Date Range or Category filter in the sidebar. | *"Every KPI and chart is completely reactive. If we filter exclusively for 'Breakfast', the entire dashboard recalculates instantly."* |
| **3. Rush Hour Trends** | Scroll to 'Sales by Hour' and 'Daily Trend'. | *"Notice how the hourly chart automatically isolates three distinct spikes: Breakfast at 9 AM, Lunch peak at 12–1 PM, and Evening tea at 5 PM. Furthermore, the weekday chart proves that Saturday and Sunday sales drop sharply."* |
| **4. Product Insights** | Switch to Tab 2: 'Product Insights'. | *"Here we compare volume velocity against revenue. Samosas generate the highest volume (nearly 2,000 units), but Veg Thalis generate the highest total revenue due to their ₹120 ticket price."* |
| **5. ML Demand Prediction** | Switch to Tab 3: 'Demand Prediction'. Select *'Masala Dosa'*. | *"Now we arrive at the core Machine Learning engine. The canteen chef selects an item, and the Random Forest model projects the exact preparation schedule for the next 7 days, highlighting peak days and expected revenue."* |
| **6. Model Architecture** | Switch to Tab 4: 'Dataset & Model Details'. | *"Everything is transparent. We show the raw dataset, feature importances, and our scikit-learn Pipeline with OneHotEncoding and chronological holdout evaluation."* |

---

## 11. Expected Evaluator Questions & High-Scoring Answers

### Q1: Why did you use Random Forest instead of Linear Regression?
**Answer:** *"Linear regression assumes a linear relationship between features and target. However, canteen demand is non-linear and step-based (for example, demand does not slowly decline on weekends; it drops abruptly because classes are closed). Random Forest splits data using decision nodes, allowing it to capture non-linear patterns, categorical interactions, and seasonal shifts much more accurately."*

### Q2: Why did you not use an LSTM or Deep Learning?
**Answer:** *"Deep learning models like LSTMs or GRUs require tens of thousands of data points to converge and are prone to overfitting on small tabular datasets. Furthermore, they are computationally heavy and act as black boxes. For a 180-day college dataset with 7 catalog items, Random Forest Regressor combined with scikit-learn's ColumnTransformer trains in under 2 seconds, achieves an R² above 0.70, and is easily explainable to kitchen managers."*

### Q3: Why is zero-sales imputation necessary?
**Answer:** *"If we only aggregate days where sales occurred, we introduce survival bias. For instance, if Lemonade is rarely sold on cold rainy days, omitting those days would trick the model into believing Lemonade sells 5 units every single day. By generating a full Cartesian grid of Dates × Items and filling zero-sale days with 0, the model learns the true baseline demand."*

### Q4: Why did you use a chronological split instead of random train/test split?
**Answer:** *"In time-series demand forecasting, a random shuffle leaks future information into past predictions. Chronological splitting (training on May–September and evaluating on October) strictly mirrors real life: predicting the future using only historical data."*

### Q5: How do you handle negative predictions?
**Answer:** *"Because food portions cannot be negative, we pass all raw model outputs through a `max(0, round(prediction))` boundary, guaranteeing non-negative integer batch counts for kitchen preparation."*

---

## 12. Project Limitations & Future Improvements
1. **Weather Integration:** Incorporating temperature and rainfall data would improve predictions for cold beverages (Cold Coffee, Lemonade) and hot snacks (Samosa).
2. **Campus Event Calendars:** Incorporating college festival dates, sports meets, and exam schedules would capture sudden demand spikes.
3. **Dynamic Buffer Stock (Quantile Regression):** Upgrading from mean regression to quantile regression (e.g., predicting the 85th percentile) to provide a safety margin against unexpected rushes.
4. **Real POS API Sync:** Integrating with a cloud billing system or QR-code ordering app for automated daily model retraining.
