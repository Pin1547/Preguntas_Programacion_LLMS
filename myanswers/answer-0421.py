from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import RidgeClassifier

def predecir_calidad_vino(df, target_col):
    # 1. Separar características y variable objetivo
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Escalar con RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Seleccionar el 50% de mejores características
    selector = SelectPercentile(
        score_func=f_classif,
        percentile=50
    )
    X_selected = selector.fit_transform(X_scaled, y)

    # 4. Entrenar RidgeClassifier
    modelo = RidgeClassifier(random_state=42)
    modelo.fit(X_selected, y)

    # 5. Devolver en el orden exacto pedido
    return modelo, scaler, selector