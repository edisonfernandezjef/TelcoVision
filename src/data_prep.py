import pandas as pd
from pycaret.classification import setup


df = pd.df = pd.read_csv("../data/raw/telco_churn.csv")
df.head()

# 2️⃣ Configurar PyCaret para limpieza
exp.setup = setup(
    data=df,
    target='churn',             # columna objetivo
    imputation_type='simple',   # imputación básica
    numeric_imputation='mean',  # reemplaza nulos numéricos por la media
    categorical_imputation='mode', # reemplaza nulos categóricos por la moda
    normalize=True,             # normaliza variables numéricas
    normalize_method='zscore',  # método estándar
    remove_outliers=False,      # (opcional: lo desactivamos para no perder filas)
    preprocess=True,            # activa el pipeline completo
    session_id=123,   # agrega reproducibilidad
    verbose=False,    # sigue funcionando en v3
    html=False        # evita generar HTML en notebooks
)
#extarer de pycaret el dataset transformado
data_transformed = exp.get_config('X')
data_transformed['churn'] = exp.get_config('y')

# 5️⃣ Guardar dataset limpio
output_path = "../data/processed/telco_churn_clean.csv"
data_transformed.to_csv(output_path, index=False)

print(f"✅ Dataset limpio guardado en: {output_path}")