import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

def detectar_piezas_defectuosas(X_train, X_test):
    # 1. Crear y ajustar el scaler solo con X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 2. Escalar X_test usando el mismo scaler
    X_test_scaled = scaler.transform(X_test)

    # 3. Crear el modelo OneClassSVM
    model = OneClassSVM(
        nu=0.1,
        kernel='rbf',
        gamma='scale'
    )

    # 4. Entrenar solo con piezas normales
    model.fit(X_train_scaled)

    # 5. Predecir sobre X_test
    preds = model.predict(X_test_scaled)

    return preds