import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

class ComparableArray:
    __array_priority__ = 10000

    def __init__(self, array):
        self.array = np.asarray(array)

    def __eq__(self, other):
        return np.array_equal(self.array, other)

    def __ne__(self, other):
        return not np.array_equal(self.array, other)

    def __array__(self):
        return self.array

def detectar_piezas_defectuosas(X_train=None, X_test=None, input=None, output=None):
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

    preds = model.predict(X_test_scaled)

    return ComparableArray(preds)