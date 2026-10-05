from src.data_loader import BASELINE_FEATURES, load_data


df = load_data()

print("Dataset shape:", df.shape)
print("\nFire distribution:")
print(df["Classes"].value_counts())
print("\nAverage conditions:")
print(df.groupby("Classes")[list(BASELINE_FEATURES)].mean())
