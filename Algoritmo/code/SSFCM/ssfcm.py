import numpy as np
from scipy.spatial import distance

def safe_distance(u, v):
    """
    Restituisce la distanza euclidea tra vettori u e v.
    Se u o v sono array 2D con shape (1, n), li riduce a 1D.
    """
    u_a = np.asarray(u)
    v_a = np.asarray(v)
    # se sono 2D con una riga, riduci a 1D
    if u_a.ndim > 1:
        if u_a.shape[0] == 1:
            u_a = u_a.ravel()
        else:
            # non dovrebbe succedere: se u è più righe prendi la prima riga
            u_a = u_a.reshape(-1)
    if v_a.ndim > 1:
        if v_a.shape[0] == 1:
            v_a = v_a.ravel()
        else:
            v_a = v_a.reshape(-1)
    return np.linalg.norm(u_a - v_a)

def initialize_centroids(data, num_of_centroids=None, n_clusters=None):
    """
    Inizializza i centroidi per FCM / ISSFCM.
    Se num_of_centroids > n_samples, permette replace per evitare ValueError.
    """
    if n_clusters is not None:
        num_of_centroids = n_clusters
    data = np.asarray(data)
    n_samples = data.shape[0]
    if num_of_centroids is None:
        raise ValueError("num_of_centroids non specificato")
    replace_flag = num_of_centroids > n_samples
    indices = np.random.choice(n_samples, num_of_centroids, replace=replace_flag)
    return data[indices]

# GET b e F (placeholder stabile)

def get_b_F(X, n_clusters=None, fuzziness_coefficient=2):
    """
    Restituisce b e F. Implementazione minimale/robusta:
    - b: vettore (n_samples,1) di ones (o altro) -- manteniamo forma compatibile
    - F: matrice identità (n_features x n_features) come placeholder
    """
    X = np.asarray(X)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    b = np.ones((n_samples, 1), dtype=float)
    F = np.eye(n_features, dtype=float)
    return b, F

# CALCOLO DEI CENTROIDI (FCM-style)

def compute_centroids(data, partition_matrix, fuzzy_value=2):
    """
    data: (n_samples, n_features)
    partition_matrix: U (n_samples, n_clusters)
    ritorna centroids shape (n_clusters, n_features)
    """
    data = np.asarray(data)
    U = np.asarray(partition_matrix)
    n_clusters = U.shape[1]
    n_features = data.shape[1]
    centroids = np.zeros((n_clusters, n_features), dtype=float)

    for k in range(n_clusters):
        um = (U[:, k] ** fuzzy_value)
        denom = np.sum(um)
        if denom == 0:
            centroids[k, :] = np.zeros(n_features)
        else:
            centroids[k, :] = (um[:, None] * data).sum(axis=0) / denom
    return centroids

# UPDATE PARTITION MATRIX (U)

def update_partition_matrix(centroids, data, dist='euclidean', b=None, F=None, a=1, fuzzy_value=2):
    """
    centroids: (n_clusters, n_features)
    data: (n_samples, n_features)
    restituisce U (n_samples, n_clusters)
    Usa safe_distance per evitare errori dimensionali.
    """
    centroids = np.asarray(centroids)
    data = np.asarray(data)
    n_samples = data.shape[0]
    n_clusters = centroids.shape[0]
    U = np.zeros((n_samples, n_clusters), dtype=float)

    for i in range(n_samples):
        # calcola distanze d_ik
        distances = np.zeros(n_clusters, dtype=float)
        for k in range(n_clusters):
            distances[k] = safe_distance(data[i], centroids[k])
        # evita zero
        distances = np.maximum(distances, 1e-12)
        # FCM update: u_ik = 1 / sum_j (d_ik / d_ij)^(2/(m-1))
        power = 2.0 / (fuzzy_value - 1.0)
        inv = (1.0 / distances) ** power
        denom = np.sum(inv)
        if denom == 0:
            # assegna uniformemente se qualcosa va storto
            U[i, :] = 1.0 / n_clusters
        else:
            U[i, :] = inv / denom

    return U

# OBJECTIVE FUNCTION

def evaluate_objective_functions(data, centroids, partition_matrix,context,
                                 fuzzy_value=2, dist='euclidean', a=None, b=None, F=None):
    """
    Versione compatibile DSSFCM/ISSFCM.
    Il parametro context è richiesto dal framework, ma può essere ignorato.
    """
    n_samples = data.shape[0]
    n_clusters = centroids.shape[0]
    J = 0.0

    for k in range(n_clusters):
        for i in range(n_samples):
            d = distance.euclidean(data[i, :], centroids[k, :])
            J += (partition_matrix[i, k] ** fuzzy_value) * (d ** 2)
    return J

