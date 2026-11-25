# Canteen Sales Regression: Short Script

About 10 minutes. Text in [SQUARE BRACKETS] means switch to the website (keep it open at http://localhost:8501) or back to the slides. The same text is in the slides' speaker notes.

## 1. Cover

We are Parv Arora and Anshika Trikha. Our project is Canteen Sales Regression: a machine learning system that predicts how many portions of each food item a college canteen should cook over the next 7 days. It is a supervised regression project, because we predict a number, the units sold. The website is just the front end. The focus today is the machine learning behind it.

## 2. The problem

Canteens lose money in two ways: cook too much and food spoils, cook too little and popular items sell out. Demand is not flat. It follows the timetable and the day of the week: about 58 units on a weekday, 27 on a Saturday and 19 on a Sunday in our test period. Most canteens plan from memory. So we ask the model one question: for this item on this date, how many units will sell? The inputs are called features (item, price, day, month) and the number we predict is the target (units sold).

## 3. The data

Where the data comes from: real canteens do not share billing data, so we generated a synthetic dataset with our own script, data/generate_data.py, using a fixed random seed (42) so it can be reproduced exactly. It covers 180 days (1 May to 27 Oct 2024), 5,970 orders and 7 menu items, with realistic patterns built in: meal-time rushes, a weekend drop and semester seasonality. There are no missing values. The log has one row per order, but the model needs one row per item per day, so we aggregate to 180 days x 7 items = 1,260 rows. Days where an item did not sell have no row, so we fill them with 0. Otherwise the model never sees a zero and over-predicts (survival bias).

[WEBSITE, 30 seconds: open the Dataset and Model Details tab, point at the counts and the records table, then come back.]

## 4. The model

How we built the model, step by step:
1. Load and validate the data.
2. Aggregate to the daily demand grid and fill missing days with 0.
3. Create features from the date: day of week, weekend flag, month, day name.
4. Split by date: first 80 percent train, last 20 percent test.
5. Build one scikit-learn Pipeline: one-hot encode the 3 label columns (item, category, day name), pass the 4 numbers through, then a Random Forest Regressor.
6. Train it on the training rows.
7. Score it on the test rows with MAE and R squared, and compare with simpler baselines.
8. Save the model. The website loads it to predict the next 7 days.

About the forest: a decision tree is a flow chart of yes/no questions, and a single tree overfits. A Random Forest trains 100 trees on different random samples of the data (bagging) and averages them, which makes it stable. Settings: 100 trees, max depth 12, at least 4 samples to split, at least 2 per leaf, seed 42. The output is clipped at 0 and rounded to whole portions. We chose it over linear regression because demand drops in steps on weekends, and over deep learning because 1,260 rows is too little data.

## 5. Training

We never test on data the model has seen. We train on the first 144 days (1 May to 21 Sep, 1,008 rows) and test on the last 36 days (22 Sep to 27 Oct, 252 rows). We split by date, not randomly, because a random shuffle would leak future days into training and inflate the score. This copies real use: learn from the past, forecast the future. Overfitting means doing well on training data but badly on new data. We limited tree depth and leaf size, and we compare train and test scores on the next slide.

## 6. Results

Two metrics, both on the unseen test days. MAE is the average miss in units: ours is 1.53, so the forecast is typically off by one or two portions. R squared is the share of demand variation the model explains: 0 means no better than guessing the average, 1 is perfect. Ours is 0.716. Baselines: overall average R2 0.00 (MAE 2.67), per-item average 0.30 (2.35), linear regression 0.62 (1.79), Random Forest 0.716 (1.53). Training R2 is 0.81, so overfitting is small.

## 7. What it learned

Feature importance shows what the forest relies on: day of week 26 percent, weekend flag 21, unit price 21 (it also identifies the item), Samosa 15, month 8. So calendar patterns drive demand. On the right, predicted vs actual daily totals on the 36 test days: the model follows the weekday highs and the weekend dips. It misses one-off events because it only uses calendar and item data.

[OPTIONAL WEBSITE, 20 seconds: Dataset and Model Details tab, Top Predictive Demand Signals.]

## 8. Live demo and close

LIVE DEMO.
[Switch to the website.]
Overview: the KPIs (Rs 5.10 lakh, 5,970 orders). Untick some categories in the sidebar and watch everything update. Show the trend, hour and weekday charts.
Product Insights: Samosa sells the most units, Veg Thali earns the most revenue.
Demand Prediction: choose Masala Dosa. The model returns 9 portions on weekdays, 5 on Saturday and 2 on Sunday, 52 in total, Rs 3,120. Change the item to show the plan updates.
[Back to the slides.]
Limitations: synthetic data, calendar-only features, one fixed split with no tuning. Next steps: weather and event data, quantile regression for a safety buffer, cross-validation, live billing data.
Thank you. Questions?

## Quick answers

**Why Random Forest?** Demand drops in steps on weekends, which trees capture and a straight line cannot. Linear regression scored 0.62 R2 against 0.716. Deep learning needs far more than 1,260 rows.
**What is a Random Forest?** Many decision trees, each trained on a different random sample of the data, with their predictions averaged.
**Why split by date?** A random shuffle leaks future days into training and inflates the score.
**Why fill missing days with 0?** No sales means no row in the log. Without zeros the model thinks every item always sells.
**Why one-hot encoding?** Numbering the items 1 to 7 would imply an order between them.
**Is the data real?** No, it is synthetic (data/generate_data.py, seed 42), because real canteen billing data is not public. Any real CSV with the same columns can be used instead.
**Is 0.716 good?** It beats every baseline we tried, and train R2 is 0.81, so overfitting is small.
