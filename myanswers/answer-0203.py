import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

def detectar_piezas_defectuosas(X_train, X_test=None):
    if isinstance(X_train, dict) and X_test is None:
        X_test = X_train["input"]["X_test"]
        X_train = X_train["input"]["X_train"]

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