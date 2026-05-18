import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def clasificar_ley_oro(df: pd.DataFrame, test_size: float):
    # 1. Crear copia del DataFrame y variable objetivo binaria
    df_proc = df.copy()
    df_proc["alta_ley"] = (df_proc["ley_oro"] >= 2.5).astype(int)

    # 2. Separar variables predictoras y objetivo
    X = df_proc.drop(columns=["ley_oro", "alta_ley"])
    y = df_proc["alta_ley"]

    # 3. Dividir entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # 4. Estandarizar variables numéricas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Entrenar modelo de clasificación
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train_scaled, y_train)

    # 6. Evaluar el modelo
    y_pred = modelo.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    matriz_conf = confusion_matrix(y_test, y_pred)

    # 7. Retornar en el mismo formato que el generador
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "matriz_confusion": matriz_conf,
        "modelo_entrenado": modelo
    }