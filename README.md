# Project-web-traffic

# Website Traffic Forecasting

A time-series forecasting project: using daily website user counts to train a **KNN Regressor** that predicts the number of users on upcoming days.

## Key Results

| Dataset | Samples | MAE | RMSE* | R² |
|---|---|---|---|---|
| Test set (last 20% of training data) | 83 | ~182 | ~251 | **0.90** |
| Unseen new data (26/08/2021 – 04/12/2021) | 92 | ~534 | ~732 | **0.79** |

\*RMSE = square root of MSE (test MSE ≈ 62,762; new-data MSE ≈ 535,215).

With an average of about 2,800 users per day, the mean error on the test set is roughly 6.5%. On new data the error rises to about 19%, but the model still explains nearly 80% of the variance. This suggests the model generalizes reasonably well, though accuracy declines as the data moves further from the training period.

## Data

- `web-traffic.csv`: 421 days (01/07/2020 – 25/08/2021) with two columns, `date` and `users`. No missing values.
- `web-traffic_new_data.csv`: 101 days (26/08/2021 – 04/12/2021), used to evaluate the model on unseen data.
- `web-traffic_report.html`: automated EDA report (ydata-profiling).

Quick stats: mean ~2,792 users/day, minimum 133, maximum 6,807.

## Methodology

1. **Exploratory analysis:** plotted the time series and generated an EDA report with ydata-profiling.
2. **Feature engineering (lag features):** used the user counts of the previous 9 days (`users_t-1` … `users_t-9`) to predict the current day (10-day sliding window).
3. **Time-based split:** first 80% for training, last 20% for testing, with `shuffle=False` to prevent future data from leaking into the training set.
4. **Model:** `KNeighborsRegressor` (`n_neighbors=15`, `weights='distance'`).
5. **Evaluation:** MAE, MSE, R².
6. **Future forecast:** recursive 100-day forecast, where each new prediction is fed back into the window to predict the next day.
7. **Model export:** saved with `pickle`, then reloaded to evaluate on the new data.

## Project Structure

```
├── web_traffic.py                  # Training, evaluation and future forecast
├── predict_new_data.py             # Load saved model, evaluate on new data
├── web_traffic_model.pkl           # Trained model
├── web-traffic.csv                 # Training data
├── web-traffic_new_data.csv        # New data for evaluation
├── web-traffic_report.html         # EDA report
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib

# 1. Train, evaluate and forecast 100 days ahead
python web_traffic.py

# 2. Use the saved model to predict on new data
python predict_new_data.py
```

Note: the `.pkl` file was created with scikit-learn 1.7.1. Use the same version (or re-run `web_traffic.py` to regenerate the model) to avoid compatibility warnings.

## Limitations and Future Improvements

- The model only uses past values and does not account for seasonality (day of week, holidays, marketing campaigns).
- A 100-day recursive forecast accumulates error, so reliability decreases for later days.
- Performance drops on new data, suggesting the model should be retrained periodically.
- Next steps: add calendar features, try Random Forest / XGBoost, ARIMA / Prophet, and use time-series cross-validation (`TimeSeriesSplit`).

## Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · ydata-profiling
