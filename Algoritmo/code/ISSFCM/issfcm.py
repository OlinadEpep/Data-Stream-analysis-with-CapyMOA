import os
import numpy as np
from SSFCM.ssfcm import initialize_centroids, compute_centroids, update_partition_matrix

# Funzione obiettivo compatibile DSSFCM

def evaluate_objective_functions(data, centroids, partition_matrix, context=None,
                                 fuzzy_value=2, dist='euclidean', a=None, b=None, F=None):
    """
    Versione compatibile DSSFCM/ISSFCM.
    Parametri extra ignorati ma richiesti da DSSFCM.
    """
    n_samples = data.shape[0]
    n_clusters = centroids.shape[0]
    J = 0.0

    for k in range(n_clusters):
        for i in range(n_samples):
            d = np.linalg.norm(data[i, :] - centroids[k, :])
            J += (partition_matrix[i, k] ** fuzzy_value) * (d ** 2)
    return J

# ISSFCM principale

def issfcm(X, number_of_clusters, fuzziness_coefficient, b, F, alpha, max_iter=100,
           stop_condition=('obj_delta', 0.001), distance='euclidean', M=None, output=None):

    # Directory dei log
    if output is None:
        output = os.path.join(os.getcwd(), "results")
    os.makedirs(output, exist_ok=True)

    samples = X.shape[0]

    # Concateno dati precedenti se M esiste
    if M is not None:
        data = np.concatenate((X, M[0]), axis=0)
        F_shape = F.shape
        M_shape = M[1].shape
        if F_shape[1] != M_shape[1]:
            temp = np.zeros((M_shape[0], F_shape[1]))
            temp[:, :M_shape[1]] = M[1]
            F = np.concatenate((F, temp), axis=0)
        else:
            F = np.concatenate((F, M[1]), axis=0)
        b = np.concatenate((b, M[2]), axis=0)
    else:
        data = X

    # Inizializzo centroidi e matrice di appartenenza
    v = initialize_centroids(data=data, num_of_centroids=number_of_clusters)
    U = update_partition_matrix(centroids=v, data=data, dist=distance, b=b, F=F, a=alpha)

    obj_functions = []
    cont = 0

    while cont < max_iter and (len(obj_functions) <= 2 or
                               np.abs(obj_functions[-1] - obj_functions[-2]) >= stop_condition[1]):
        v = compute_centroids(partition_matrix=U, data=data, fuzzy_value=fuzziness_coefficient)
        U = update_partition_matrix(centroids=v, data=data, dist=distance, b=b, F=F, a=alpha)
        obj_f_value = evaluate_objective_functions(
            data=data, centroids=v, partition_matrix=U, context=None,
            fuzzy_value=fuzziness_coefficient, dist=distance, a=alpha, b=b, F=F
        )
        obj_functions.append(obj_f_value)
        cont += 1

    # Salva log
    outputFile = os.path.join(output, "Log.txt")
    with open(outputFile, "a") as f:
        f.write(f"Chunk Data (shape={data.shape})\n")
        np.savetxt(fname=f, X=data, fmt="%2.4f", delimiter=",", footer="--------")
        f.write("\nPrototypes\n")
        np.savetxt(fname=f, X=v, fmt="%2.4f", delimiter=",", footer="--------")
        f.write("\nMembership Matrix\n")
        np.savetxt(fname=f, X=U, fmt="%2.4f", delimiter=",")
        f.write("\n" + "-"*100 + "\n")

    return U[:samples, ], v

# Funzione per creare la memoria

def buildMemory(prototipi):
    num_prot = prototipi.shape[0]
    Fv = np.zeros((num_prot, num_prot), dtype=float)
    for i in range(num_prot):
        label = int(prototipi[i, -1])
        if label != -1:
            Fv[i, label] = 1

    b1 = np.zeros((num_prot, 1), dtype=float)
    for i in range(num_prot):
        label = int(round(prototipi[i, -1]))
        b1[i] = 0 if label == -1 else 1

    return (prototipi, Fv, b1)
