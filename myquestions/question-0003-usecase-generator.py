import numpy as np

def generar_caso_de_uso_isomap_kmeans_clustering():
    from sklearn.datasets import make_s_curve
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import Isomap
    from sklearn.cluster import KMeans

    X, _ = make_s_curve(
        n_samples=25,
        noise=0.1,
        random_state=np.random.randint(100)
    )
    n_neigh = 5
    clusters = np.random.randint(2, 5)

    # Cálculo real de la salida esperada
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    isomap = Isomap(n_neighbors=n_neigh, n_components=2)
    X_iso = isomap.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_iso)

    params = {
        "X": X,
        "n_neighbors_isomap": n_neigh,
        "n_clusters": clusters
    }

    esperado = (X_iso, labels)

    print("=== PARÁMETROS DE ENTRADA ===")
    print(f"n_neighbors_isomap: {params['n_neighbors_isomap']}")
    print(f"n_clusters: {params['n_clusters']}")
    print(f"Shape de X: {params['X'].shape}")
    print("\nX de entrada:")
    print(params["X"])

    print("\n=== RESPUESTA ESPERADA ===")
    print("Coordenadas Isomap:")
    print(esperado[0])
    print("\nEtiquetas de clúster:")
    print(esperado[1])

    return params, esperado