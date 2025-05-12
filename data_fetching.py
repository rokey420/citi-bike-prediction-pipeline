import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("citi_bike_hourly_top3_final", version=1)
df = fg.read()
df.to_csv("citi_bike_hourly_top3.csv", index=False)
print("Data downloaded and saved.")
