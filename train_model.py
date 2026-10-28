import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).parent
DATA = BASE / "data" / "mpg.csv"
MODEL_PATH = BASE / "models" / "rf_mpg.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"

NUM_FEATURES = ["cylinders", "displacement", "horsepower", "weight_kg", "acceleration", "year"]
CAT_FEATURES = ["origin"]
TARGET = "mpg"

df = pd.read_csv(DATA).rename(columns={"model_year": "year"})
if df["year"].max() < 100:
    df["year"] += 1900
df["weight_kg"] = df["weight"] * 0.45359237
df = df.dropna(subset=[TARGET])  # on garde les NaN des features (imputés ensuite)

X = df[NUM_FEATURES + CAT_FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
])
model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
metrics = {
    "MAE": float(mean_absolute_error(y_test, pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
    "R2": float(r2_score(y_test, pred)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)
METRICS_PATH.write_text(json.dumps(metrics, indent=2))
print("Modèle sauvegardé :", MODEL_PATH)
print(json.dumps(metrics, indent=2))