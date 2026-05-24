import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load Dataset

df = pd.read_csv("dataset/weather_dataset.csv")

# Features

X = df[[
    "Temp_C",
    "Rel Hum_%",
    "Wind Speed_km/h",
    "Visibility_km",
    "Press_kPa"
]]

# Target

y = df["Weather"]

# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model

model = RandomForestClassifier()

# Train Model

model.fit(X_train, y_train)

# Save Model

pickle.dump(model, open("model.pkl", "wb"))

print("Model Trained Successfully")
