import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def calcular_compacidad_clusters(X, n_clusters, random_state=42):
    # 1. Escalar X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Entrenar KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_state
    )
    kmeans.fit(X_scaled)

    # 3. Obtener etiquetas y centroides
    etiquetas = kmeans.labels_
    centroides = kmeans.cluster_centers_

    # 4. Calcular distancia euclidiana de cada muestra a su centroide asignado
    distancias = np.linalg.norm(
        X_scaled - centroides[etiquetas],
        axis=1
    )

    # 5. Devolver la distancia promedio como float
    compacidad = float(np.mean(distancias))

    return compacidad