# cluster_splitting.py
from CFCM.cfcm import cfcm, update_partition_matrix
import numpy as np
from scipy.linalg import norm
import pandas as pd

def split_cluster(data, context, C=2, number_of_clusters=2):
    """
    Esegue lo split di un cluster utilizzando la colonna della matrice di membership come contesto.
    Restituisce la nuova matrice di membership U e i centroidi v.
    """
    indices = []
    features = np.shape(data)[1]
    slice_data = np.zeros((0, features))

    # Seleziona i dati che superano la soglia 1/C
    for i, d in enumerate(data):
        if context[i] > 1 / C:
            slice_data = np.concatenate((slice_data, d.reshape(1, features)))
            indices.append(i)

    if slice_data.shape[0] == 0:
        print("⚠️ Nessun dato supera la soglia 1/C")
        return None, None

    try:
        # Chiama cfcm e salva tutto in una variabile
        result = cfcm(
            X=slice_data,
            context=context[indices],
            number_of_clusters=number_of_clusters,
            fuzziness_coefficient=2
        )
        print("Debug cfcm result:", result)  # Debug: vedere struttura

        # Gestione flessibile dell'unpack
        if isinstance(result, (tuple, list)):
            if len(result) >= 2:
                U_chunk = result[0]
                v = result[1]
            else:
                raise ValueError("cfcm non restituisce abbastanza valori")
        else:
            raise TypeError("cfcm restituisce un tipo non previsto")

        # Aggiorna la matrice di appartenenza completa per tutti i dati
        U_full = update_partition_matrix(
            centroids=v,
            data=data,
            context=context,
            fuzzy_value=2,
            dist='euclidean'
        )

        return U_full, v

    except Exception as e:
        print(f"⚠️ Errore nello split del cluster: {e}")
        return None, None


def replace_cluster(data, U, v, cluster, new_U, new_v, map_cluster_class):
    idx_cluster = int(cluster)
    label = v[idx_cluster, -1]
    samples = len(list(data[data[:, -1] == label]))
    print(f"Cluster {label} {samples} samples")

    for i, d in enumerate(data):
        if d[-1] == label:
            a = np.argmax(new_U[i, :])
            class_ass = label if a == 0 else max(map_cluster_class.keys()) + 1
            d[-1] = class_ass

    first = True
    class_ass = map_cluster_class[label]

    for index, p in enumerate(new_v):
        if first:
            U[:, idx_cluster] = new_U[:, index]
            p[-1] = label
            v[idx_cluster, :] = p
            first = False
        else:
            label = max(map_cluster_class.keys()) + 1
            p[-1] = label
            map_cluster_class[label] = class_ass
            v = np.append(v, p.reshape((1, np.shape(p)[0])), axis=0)
            col = new_U[:, index]
            U = np.append(U, col.reshape(np.shape(col)[0], 1), axis=1)

    return data, U, v, map_cluster_class


def select_cluster(X, U, v, crit, path_result=None, maximize=True, num_chunk=0):
    prototypes_costs = {}
    max_cost = 0.0
    cluster_to_split = -1

    # Calcolo dell'errore per ogni cluster
    for i, k in enumerate(list(v[:, -1])):
        prototype_cost = crit(X, U, v, i)
        prototypes_costs[round(k)] = prototype_cost

        if maximize and prototype_cost > max_cost:
            max_cost = prototype_cost
            cluster_to_split = round(k)

    # Salvataggio risultati in modo leggibile
    if path_result is not None:
        file_path = f"{path_result}/rec_error.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n" + "─" * 50 + "\n")
            f.write(f"📦 CHUNK {num_chunk}\n")
            f.write("─" * 50 + "\n")
            f.write(f"🔹 Cluster analizzati: {len(prototypes_costs)}\n")
            for cid, err in prototypes_costs.items():
                f.write(f"   • Cluster {cid} → Reconstruction Error: {err:,.2f}\n")
            f.write("\n")
            f.write(f"📈 Cluster selezionato per lo split → {cluster_to_split}\n")
            f.write(f"💡 Motivazione: errore più alto ({max_cost:,.2f})\n")
            f.write("─" * 50 + "\n")

    return max_cost, cluster_to_split


def reconstruction_error(X, U, v, idx_cluster):
    """Calcola l'errore di ricostruzione per il cluster specificato."""
    data_recnstrdata_distance_summ = 0.0
    q = norm(X, ord='fro')

    for i, sample in enumerate(X):
        memb_vector_u = U[i, :]
        if np.argmax(memb_vector_u) == idx_cluster:
            recnstrctd_sample = reconstruct_sample(memb_vector_u, v)
            d = np.linalg.norm(np.ravel(sample) - np.ravel(recnstrctd_sample))
            data_recnstrdata_distance_summ += d ** 2

    return data_recnstrdata_distance_summ / q


def reconstruct_sample(u, v):
    nr_clusters = v.shape[0]
    numerator_summ = 0.0
    denominator_summ = 0.0

    for k in range(nr_clusters):
        numerator_summ += pow(u[k], 2) * v[k, :]
        denominator_summ += pow(u[k], 2)

    return numerator_summ / denominator_summ
