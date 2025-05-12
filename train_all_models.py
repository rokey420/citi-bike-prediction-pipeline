import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

# Read the downloaded dataset
df = pd.read_csv("citi_bike_hourly_top3.csv")
df = df.sort_values(by=["start_station_id", "pickup_hour"])

# -------------------- Model 1: Baseline (rides(t) = rides(t-1)) --------------------
mlflow.set_experiment("baseline_model_experiment")
with mlflow.start_run(run_name="Model_1_Baseline"):

    df["predicted_rides"] = df.groupby("start_station_id")["rides"].shift(1)
    baseline_df = df.dropna()
    mae1 = mean_absolute_error(baseline_df["rides"], baseline_df["predicted_rides"])
    print(f"Baseline MAE: {mae1:.3f}")

    # Log results
    mlflow.log_metric("baseline_mae", mae1)
    baseline_df[["pickup_hour", "start_station_id", "rides", "predicted_rides"]].to_csv("baseline_predictions.csv", index=False)
    mlflow.log_artifact("baseline_predictions.csv")
    mlflow.set_tag("model_type", "baseline")


# -------------------- Model 2: LightGBM with 28 lag features --------------------
mlflow.set_experiment("lag_model_experiment")
with mlflow.start_run(run_name="Model_2_Lag28"):

    # Create lag features
    for lag in range(1, 29):
        df[f"lag_{lag}"] = df.groupby("start_station_id")["rides"].shift(lag)
    df2 = df.dropna()

    feature_cols = [f"lag_{i}" for i in range(1, 29)]
    X = df2[feature_cols]
    y = df2["rides"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model2 = lgb.LGBMRegressor()
    model2.fit(X_train, y_train)

    preds2 = model2.predict(X_test)
    mae2 = mean_absolute_error(y_test, preds2)
    print(f"Model 2 MAE: {mae2:.3f}")

    # Log to MLflow
    mlflow.log_metric("lag28_mae", mae2)
    joblib.dump(model2, "lag_model.pkl")
    mlflow.log_artifact("lag_model.pkl")
    mlflow.set_tag("model_type", "lightgbm_lag28")


# -------------------- Model 3: LightGBM with top 10 features --------------------
mlflow.set_experiment("top10_model_experiment")
with mlflow.start_run(run_name="Model_3_Top10"):

    # Get top 10 features
    importances = pd.Series(model2.feature_importances_, index=X_train.columns)
    top10_features = importances.sort_values(ascending=False).head(10).index.tolist()

    X_train_reduced = X_train[top10_features]
    X_test_reduced = X_test[top10_features]

    model3 = lgb.LGBMRegressor()
    model3.fit(X_train_reduced, y_train)
    preds3 = model3.predict(X_test_reduced)

    mae3 = mean_absolute_error(y_test, preds3)
    print(f"Model 3 MAE (top 10): {mae3:.3f}")

    # Log to MLflow
    mlflow.log_metric("top10_mae", mae3)
    joblib.dump(model3, "top10_model.pkl")
    mlflow.log_artifact("top10_model.pkl")
    mlflow.set_tag("model_type", "lightgbm_top10")
