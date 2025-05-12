import os
import hopsworks

# Load credentials from environment variables
api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT")

if api_key is None or project_name is None:
    raise ValueError("Missing HOPSWORKS_API_KEY or HOPSWORKS_PROJECT environment variables.")

# Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key, project=project_name)

# Fetch feature group data
fs = project.get_feature_store()
fg = fs.get_feature_group("citi_bike_hourly_top3_final", version=1)
df = fg.read()

# Save as local CSV
df.to_csv("citi_bike_hourly_top3.csv", index=False)
print("✅ Data downloaded and saved to citi_bike_hourly_top3.csv")
