import pandas as pd
import numpy as np

def generar_caso_de_uso_analizar_desigualdad_salarial():
    deptos = ["Ventas", "Ingeniería", "Marketing", "Soporte"]
    data = []
    
    for d in deptos:
        # Generar salarios con diferentes niveles de dispersión
        mean = np.random.randint(3000, 6000)
        std = np.random.randint(100, 2000)
        salarios = np.clip(np.random.normal(mean, std, 10), 1500, None)
        
        for s in salarios:
            data.append({"depto": d, "sueldo": round(float(s), 2)})
            
    df = pd.DataFrame(data)
    umbral = 0.25
    
    # Cálculo interno
    stats = df.groupby("depto")["sueldo"].agg(["mean", "std"])
    stats["cv"] = stats["std"] / stats["mean"]
    
    res = (
        stats[stats["cv"] >= umbral]
        .reset_index()[["depto", "mean", "cv"]]
        .rename(columns={"mean": "mean_salario", "cv": "cv_salario"})
        .sort_values("cv_salario", ascending=False)
        .reset_index(drop=True)
    )

    params = {
        "df": df,
        "depto_col": "depto",
        "salario_col": "sueldo",
        "umbral_cv": umbral
    }

    print("=== PARÁMETROS DE ENTRADA ===")
    print(f"depto_col: {params['depto_col']}")
    print(f"salario_col: {params['salario_col']}")
    print(f"umbral_cv: {params['umbral_cv']}")
    print("\nDataFrame de entrada:")
    print(params["df"])

    print("\n=== RESPUESTA ESPERADA ===")
    print(res)

    return params, res