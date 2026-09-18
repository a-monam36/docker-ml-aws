import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([
    [2000, 3, 10], 
    [1500, 2, 5], 
    [3000, 4, 15], 
    [1200, 1, 2]
])
y = np.array([500000, 350000, 750000, 250000])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.joblib")