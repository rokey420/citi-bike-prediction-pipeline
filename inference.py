import pandas as pd
import joblib

# Step 1: Load the latest input data
df = pd.read_csv("citi_bike_hourly_top3.csv")
df = df.sort_values(by=["start_station_id", "pickup_hour"])

# Step 2: Recreate lag-28 features (same as training)
for lag in range(1, 29):
    df[f"lag_{lag}"] = df.groupby("start_station_id")["rides"].shift(lag)

# Step 3: Drop rows with missing lags
df.dropna(inplace=True)

# Step 4: Prepare features
X = df[[f"lag_{i}" for i in range(1, 29)]]

# Step 5: Load the trained model (make sure this path is correct)
model = joblib.load("lag_model.pkl")

# Step 6: Predict
preds = model.predict(X)

# Step 7: Save output
df_output = df[["pickup_hour", "start_station_id", "rides"]].copy()
df_output["predicted_rides"] = preds
df_output.to_csv("inference_output.csv", index=False)

print("Inference complete. Saved to inference_output.csv")
