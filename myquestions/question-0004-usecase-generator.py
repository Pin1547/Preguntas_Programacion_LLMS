import numpy as np

def generar_caso_de_uso_evaluar_robustez_loocv():
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import LeaveOneOut, cross_val_score

    n_samples = np.random.randint(8, 15)
    X = np.random.rand(n_samples, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, n_samples)
    tipo = np.random.choice(['linear', 'ridge'])

    # Instanciar modelo esperado
    if tipo == 'linear':
        modelo = LinearRegression()
    elif tipo == 'ridge':
        modelo = Ridge(alpha=1.0)
    else:
        raise ValueError("modelo_tipo debe ser 'linear' o 'ridge'")

    # LOOCV
    loo = LeaveOneOut()

    # cross_val_score devuelve MSE negativo
    scores = cross_val_score(
        modelo,
        X,
        y,
        cv=loo,
        scoring='neg_mean_squared_error'
    )

    mse_promedio = float(-scores.mean())

    params = {
        "X": X,
        "y": y,
        "modelo_tipo": tipo
    }

    print("=== PARÁMETROS DE ENTRADA ===")
    print(f"modelo_tipo: {params['modelo_tipo']}")
    print(f"Shape de X: {params['X'].shape}")
    print(f"Shape de y: {params['y'].shape}")
    print("\nX de entrada:")
    print(params["X"])
    print("\ny de entrada:")
    print(params["y"])

    print("\n=== RESPUESTA ESPERADA ===")
    print(mse_promedio)

    return params, mse_promedio