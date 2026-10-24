import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

# Training dataset: 20 realistic property records
# Features: [sqft (float/int), bedrooms (int), age (int)]
# Target: price (USD)
X = np.array([
    [950, 1, 30],
    [1100, 2, 25],
    [1250, 2, 20],
    [1400, 3, 18],
    [1550, 3, 15],
    [1700, 3, 12],
    [1850, 3, 10],
    [2000, 3, 8],
    [2150, 4, 14],
    [2300, 4, 11],
    [2450, 4, 7],
    [2600, 4, 5],
    [2800, 4, 3],
    [3000, 5, 6],
    [3200, 5, 4],
    [3400, 5, 2],
    [3600, 5, 1],
    [1300, 2, 35],
    [1900, 4, 22],
    [2700, 4, 16],
], dtype=float)

y = np.array([
    215000,
    265000,
    305000,
    345000,
    385000,
    420000,
    455000,
    495000,
    515000,
    555000,
    595000,
    640000,
    690000,
    735000,
    785000,
    835000,
    890000,
    280000,
    450000,
    610000,
], dtype=float)

model = LinearRegression()
model.fit(X, y)

# Save serialized model artifact
joblib.dump(model, "model.joblib")
print(f"Model successfully trained on {len(X)} records and saved to model.joblib.")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficients [sqft, bedrooms, age]: {model.coef_}")

print(f"Model successfully trained on {len(X)} records and saved to model.joblib.")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficients [sqft, bedrooms, age]: {model.coef_}")
joblib.dump(model, "model.joblib")