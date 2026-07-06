import numpy as np
from scipy.spatial import distance

# INIZIALIZZAZIONE CENTROIDI

def initialize_centroids(n_features, n_clusters):
    """
    Inizializza i centroidi a zeri.
    """
    return np.zeros((n_clusters, n_features), dtype=float)

# INIZIALIZZAZIONE MATRICE DI APPARTENENZA

def initialize_partition_matrix(n_samples, n_clusters):
    """
    Inizializza la matrice di appartenenza fuzzy U.
    Ogni riga viene normalizzata a 1.
    """
    U = np.random.rand(n_samples, n_clusters)
    U = U / np.sum(U, axis=1, keepdims=True)
    return U

# CALCOLO CENTROIDI

def compute_centroids(partition_matrix, data, fuzzy_value):
    """
    Calcola i centroidi pesati secondo la matrice di appartenenza fuzzy.
    """
    n_samples, n_features = data.shape
    n_clusters = partition_matrix.shape[1]
    centroids = np.zeros((n_clusters, n_features))

    for c in range(n_clusters):
        um = partition_matrix[:, c] ** fuzzy_value
        numerator = np.dot(um, data)
        denominator = np.sum(um)
        if denominator == 0:
            centroids[c] = np.zeros(n_features)
        else:
            centroids[c] = numerator / denominator
    return centroids

# AGGIORNAMENTO MATRICE U (con contesto)

def update_partition_matrix(centroids, data, context, fuzzy_value=2, dist='euclidean'):
    """
    Aggiorna la matrice di appartenenza fuzzy U considerando anche il contesto.
    Il contesto modula il grado di appartenenza di ciascun campione.
    """
    n_samples = data.shape[0]
    n_clusters = centroids.shape[0]
    U = np.zeros((n_samples, n_clusters))
    m = fuzzy_value

    for i in range(n_samples):
        x = data[i, :]  # 1-D
        c_i = context[i] if context is not None and len(context) == n_samples else 1.0

        # calcola distanze 1-D
        distances = np.array([
            distance.euclidean(x, centroids[k, :]) for k in range(n_clusters)
        ])
        distances = np.maximum(distances, 1e-10)  # evita zero
        inv_d = 1.0 / distances
        base_u = inv_d / np.sum(inv_d)

        # il contesto pesa la membership
        U[i, :] = c_i * base_u

    # normalizza righe (sommatoria = 1)
    U = U / np.sum(U, axis=1, keepdims=True)
    return U

# FUNZIONE OBIETTIVO

def evaluate_objective_functions(data, centroids, partition_matrix, context, fuzzy_value=2, dist='euclidean'):
    """
    Calcola la funzione obiettivo del CFCM.
    """
    n_samples = data.shape[0]
    n_clusters = centroids.shape[0]
    J = 0.0

    for k in range(n_clusters):
        for i in range(n_samples):
            d = distance.euclidean(data[i, :], centroids[k, :])
            c_i = context[i] if context is not None and len(context) == n_samples else 1.0
            J += c_i * ((partition_matrix[i, k] ** fuzzy_value) * (d ** 2))
    return J

# ALGORITMO PRINCIPALE CFCM

def cfcm(X, context, number_of_clusters, fuzziness_coefficient, 
         max_iter=100, stop_condition=('obj_delta', 0.001), distance='euclidean'):
    cardinality_samples = np.shape(X)[0]
    number_features = np.shape(X)[1]
    v = initialize_centroids(n_features=number_features, n_clusters=number_of_clusters)
    U = initialize_partition_matrix(n_samples=cardinality_samples, n_clusters=number_of_clusters)
    obj_functions = []
    cont = 0

    while cont < max_iter and (len(obj_functions) <= 2 or abs(obj_functions[-1] - obj_functions[-2]) >= stop_condition[1]):
        v = compute_centroids(partition_matrix=U, data=X, fuzzy_value=fuzziness_coefficient)
        U = update_partition_matrix(centroids=v, data=X, context=context, fuzzy_value=fuzziness_coefficient, dist=distance)
        obj_f_value = evaluate_objective_functions(data=X, centroids=v, partition_matrix=U, context=context, fuzzy_value=fuzziness_coefficient, dist=distance)
        obj_functions.append(obj_f_value)
        cont += 1

    # 🔹 RITORNA SEMPRE TRE VALORI
    return U, v, obj_functions[-1]

