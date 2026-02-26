import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ================= LOAD DATA =================
df1 = pd.read_csv("city_day.csv")
# df2 = pd.read_csv("city_hour.csv")
# df3 = pd.read_csv("station_day.csv")
# df4 = pd.read_csv("station_hour.csv")
# df5 = pd.read_csv("stations.csv")

# Combine safely
# df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

print("Combined shape:", df1.shape)

# ================= CLEANING =================

# Keep only numeric columns for ML
df_numeric = df1.select_dtypes(include=[np.number])

# Fill missing values
df_numeric = df_numeric.fillna(df_numeric.mean())

print("\nMissing values after cleaning:")
print(df_numeric.isnull().sum())

# ================= HEATMAP =================
plt.figure(figsize=(6,4))
sns.heatmap(df_numeric.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
# plt.grid
# plt.legend
plt.show()

# ================= TARGET =================
target_column = "PM2.5"

if target_column not in df_numeric.columns:
    raise ValueError(f"❌ Target column '{target_column}' not found!")

X = df_numeric.drop(columns=[target_column])
y = df_numeric[target_column]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= MODELS =================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ================= EVALUATION FUNCTION =================
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n🔹 {model_name}")
    print("MAE :", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2  :", round(r2, 3))

# ✅ Correct calls (outside function)
evaluate_model(y_test, lr_pred, "Linear Regression")
evaluate_model(y_test, dt_pred, "Decision Tree")
evaluate_model(y_test, rf_pred, "Random Forest")

# ================= FEATURE IMPORTANCE =================
importances = rf_model.feature_importances_
feature_names = X.columns

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(8,4))
feat_imp.head(10).plot(kind="bar")
plt.title("Top Feature Importance - Random Forest")
plt.ylabel("Importance Score")
plt.show()

# ================= PREDICTION =================
# IMPORTANT: columns must match training features
new_data = pd.DataFrame([X.mean()])  # safe dummy example
prediction = rf_model.predict(new_data)

print("\nPredicted Pollutant Concentration:", prediction[0])

# ================= SAVE MODEL =================
joblib.dump(rf_model, "air_quality_model.pkl")
print("✅ Model saved successfully as air_quality_model.pkl")