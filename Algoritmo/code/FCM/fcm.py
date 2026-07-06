import numpy as np
from scipy.spatial import distance
import random

# INIZIALIZZAZIONE CENTROIDI

def initialize_centroids(n_features, n_clusters):
    """
    Inizializza i centroidi con zeri.
    """
    return np.zeros((n_clusters, n_features), dtype=float)

# INIZIALIZZAZIONE MATRICE DI APPARTENENZA

def initialize_partition_matrix(n_samples, n_clusters):
    """
    Inizializza casualmente la matrice di appartenenza fuzzy U.
    Ogni riga viene normalizzata a 1.
    """
    U = np.random.rand(n_samples, n_clusters)
    return U / np.sum(U, axis=1, keepdims=True)

# CALCOLO CENTROIDI

def compute_centroids(partition_matrix, data, fuzzy_value):
    """
    Aggiorna i centroidi v in base alla matrice di appartenenza U.
    """
    n_samples, n_features = data.shape
    n_clusters = partition_matrix.shape[1]
    centroids = np.zeros((n_clusters, n_features))

    for c in range(n_clusters):
        um = partition_matrix[:, c] ** fuzzy_value
        denom = np.sum(um)
        centroids[c] = np.dot(um, data) / denom if denom != 0 else np.zeros(n_features)
    return centroids

# AGGIORNAMENTO MATRICE U

def update_partition_matrix(centroids, data, fuzzy_value=2, dist='euclidean'):
    """
    Aggiorna la matrice di appartenenza fuzzy U.
    """
    n_samples = data.shape[0]
    n_clusters = centroids.shape[0]
    U = np.zeros((n_samples, n_clusters))

    for i in range(n_samples):
        x = data[i]
        distances = np.array([distance.euclidean(x, centroids[k]) for k in range(n_clusters)])
        distances = np.maximum(distances, 1e-10)  # evita divisione per zero
        inv_d = 1.0 / distances
        U[i] = inv_d / np.sum(inv_d)
    return U

# FUNZIONE OBIETTIVO

def evaluate_objective_functions(data, centroids, partition_matrix, context=None,
                                 fuzzy_value=2, dist='euclidean', a=None, b=None, F=None):
    """
    Versione compatibile DSSFCM/ISSFCM.
    Parametri extra ignorati ma necessari.
    """
    n_samples = data.shape[0]
    n_clusters = centroids.shape[0]
    J = 0.0

    for k in range(n_clusters):
        for i in range(n_samples):
            d = distance.euclidean(data[i], centroids[k])
            J += (partition_matrix[i, k] ** fuzzy_value) * (d ** 2)
    return J

# ALGORITMO PRINCIPALE FCM

def fcm(X, number_of_clusters, fuzziness_coefficient=2, max_iter=100, stop_condition=('obj_delta', 0.001)):
    """
    Esegue il Fuzzy C-Means standard compatibile DSSFCM/ISSFCM.
    """
    n_samples, n_features = X.shape
    v = initialize_centroids(n_features, number_of_clusters)
    U = initialize_partition_matrix(n_samples, number_of_clusters)
    obj_functions = []

    for iteration in range(max_iter):
        # Step 1: aggiorna i centroidi
        v = compute_centroids(U, X, fuzziness_coefficient)

        # Step 2: aggiorna la matrice U
        U = update_partition_matrix(v, X, fuzzy_value=fuzziness_coefficient)

        # Step 3: valuta la funzione obiettivo
        J = evaluate_objective_functions(X, v, U, context=None, fuzzy_value=fuzziness_coefficient)
        obj_functions.append(J)

        # Step 4: condizione di stop
        if iteration > 1 and abs(obj_functions[-1] - obj_functions[-2]) < stop_condition[1]:
            break

    return U, v, obj_functions[-1]
