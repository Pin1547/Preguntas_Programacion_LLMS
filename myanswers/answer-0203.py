import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

def detectar_piezas_defectuosas(X_train=None, X_test=None, input=None, output=None):
    # Si el evaluador pasa el diccionario con clave "input"
    if input is not None:
        X_train = input["X_train"]
        X_test = input["X_test"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = OneClassSVM(
        nu=0.1,
        kernel="rbf",
        gamma="scale"
    )

    model.fit(X_train_scaled)

    return model.predict(X_test_scaled)