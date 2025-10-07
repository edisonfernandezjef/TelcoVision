import pandas as pd
import json
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# 1️⃣ Leer parámetros desde params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

C = params["model"]["params"]["C"]
max_iter = params["model"]["params"]["max_iter"]
test_size = params["split"]["test_size"]
random_state = params["split"]["random_state"]
model_path = params["path"]["model_path"]
metrics_path = params["path"]["metrics_path"]

# 2️⃣ Cargar dataset limpio
df = pd.read_csv("data/processed/telco_churn_clean.csv")

# 3️⃣ Separar features y target
X = df.drop("churn", axis=1)
y = df["churn"]

# 4️⃣ Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# 5️⃣ Entrenar modelo
model = LogisticRegression(C=C, max_iter=max_iter)
model.fit(X_train, y_train)

# 6️⃣ Evaluar modelo
y_pred = model.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}

# 7️⃣ Guardar métricas
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

# 8️⃣ Guardar modelo entrenado
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)

print(f"✅ Modelo guardado en: {model_path}")
print(f"📊 Métricas guardadas en: {metrics_path}")
print(json.dumps(metrics, indent=4))
