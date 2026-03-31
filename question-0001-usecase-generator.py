import pandas as pd
import numpy as np

def generar_caso_de_uso_detectar_rafagas_sensores():
    n_sensores = np.random.randint(2, 5)
    sensores = [f"Sensor_{i}" for i in range(n_sensores)]
    data = []

    for s in sensores:
        tiempo_base = pd.Timestamp("2026-03-30 08:00:00")
        n_registros = np.random.randint(4, 8)

        for _ in range(n_registros):
            # Mezclamos intervalos cortos y largos
            segundos = int(np.random.choice([1, 2, 3, 6, 10, 15]))
            tiempo_base += pd.Timedelta(seconds=segundos)

            data.append({
                "sensor_id": s,
                "ts": tiempo_base
            })

    # Desordenamos filas para obligar a restaurar el orden original
    df = pd.DataFrame(data).sample(frac=1).reset_index(drop=True)

    # Opcional: descomenta esto si quieres probar la conversión a datetime
    # df["ts"] = df["ts"].astype(str)

    margen = 5

    # Construcción de la salida esperada
    df_res = df.copy()
    df_res["_orden_original"] = np.arange(len(df_res))

    df_res["ts"] = pd.to_datetime(df_res["ts"])
    df_res = df_res.sort_values(["sensor_id", "ts"])

    diff = df_res.groupby("sensor_id")["ts"].diff().dt.total_seconds()
    df_res["es_rafaga"] = (diff < margen).fillna(False)

    output = df_res.sort_values("_orden_original").drop(columns="_orden_original")

    params = {
        "df": df,
        "sensor_col": "sensor_id",
        "tiempo_col": "ts",
        "margen_segundos": margen
    }

    print("=== PARÁMETROS DE ENTRADA ===")
    print(f"sensor_col: {params['sensor_col']}")
    print(f"tiempo_col: {params['tiempo_col']}")
    print(f"margen_segundos: {params['margen_segundos']}")
    print("\nDataFrame de entrada:")
    print(params["df"])

    print("\n=== RESPUESTA ESPERADA ===")
    print(output)

    return params, output