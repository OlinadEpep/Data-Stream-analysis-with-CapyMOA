import os
import numpy as np
import pandas as pd
from CFCM.cluster_splitting import select_cluster, reconstruction_error, split_cluster, replace_cluster
from FCM.fcm import fcm
from ISSFCM.issfcm import issfcm, buildMemory
from SSFCM.ssfcm import get_b_F
from utility.utilities import distinct, create_clustering
from utility.test_utilities import plot_clusters_new

# Funzione per fare il mapping tra cluster e classi + ricostruzione dataset
def mapClusters(data, fuzziness_coefficient, map_cluster_classi, n_cluster_per_class):
    classes = distinct(data[:, -1])
    rebuilt_dataset = np.zeros(shape=(0, np.shape(data)[1]), dtype=float)
    if map_cluster_classi is None:
        map_cluster_classi = {}
    keys = map_cluster_classi.keys()
    cluster_builder = len(keys)
    for c in classes:
        if c not in list(map_cluster_classi.values()):
            for i in range(n_cluster_per_class):
                map_cluster_classi[cluster_builder + i] = c
                cluster_builder += n_cluster_per_class

    clusters_per_class = {}
    for c in classes:
        clusters_per_class[c] = list(map_cluster_classi.values()).count(c)

    for c in classes:
        temp = data[data[:, -1] == c]
        if np.shape(temp)[0] == 1:
            rebuilt_dataset = np.concatenate((rebuilt_dataset, temp), axis=0)
        else:
            temp_set = [x for x in map_cluster_classi.keys() if map_cluster_classi[x] == c]
            membership_matrix, prototypes = fcm(temp, clusters_per_class[c], fuzziness_coefficient)
            a = np.argmax(membership_matrix, axis=1)
            for i in range(np.shape(temp)[0]):
                temp[i, -1] = temp_set[a[i]]
            rebuilt_dataset = np.concatenate((rebuilt_dataset, temp), axis=0)

    if list(data[:, -1]).count(-1) > 0:
        temp = data[data[:, -1] == -1]
        rebuilt_dataset = np.concatenate((rebuilt_dataset, temp), axis=0)

    return rebuilt_dataset, map_cluster_classi

# DISSFCM con split

def dissfcm(X, fuzziness_coefficient, max_iter=100, alpha=1.0,
            stop_condition=('obj_delta', 0.001), distance='euclidean', M=None, map_cluster_classi=None,
            cluster_per_class=1, V=0, E=0, output=None, num_chunk=0, vMaxList=None):

    os.makedirs(output, exist_ok=True)
    if vMaxList is None:
        vMaxList = []

    train, map_cluster_classi = mapClusters(X, fuzziness_coefficient, map_cluster_classi, cluster_per_class)
    n_clusters = len(map_cluster_classi)

    b, F = get_b_F(X=train, n_clusters=n_clusters)
    membership_matrix, prototypes = issfcm(X=train, number_of_clusters=n_clusters,
                                           fuzziness_coefficient=fuzziness_coefficient, b=b, F=F, alpha=alpha,
                                           max_iter=max_iter, stop_condition=stop_condition, distance=distance,
                                           M=M, output=output)

    if output is not None:
        data_clustered = create_clustering(X=train, U=membership_matrix)
        t_c = "DISSFCM pre-split Chunk " + str(num_chunk)
        plot_clusters_new(n=0, centroids=prototypes, clusters=data_clustered, map=map_cluster_classi,
                          path_result=output, namefile_result="DISSFCM_chunk" + str(num_chunk),
                          title_image=t_c)

    cost, cluster_2_split = select_cluster(X=train, U=membership_matrix,
                                           v=prototypes, crit=reconstruction_error,
                                           path_result=output, maximize=True)

    vMaxList.append((cost, "", len(prototypes)))
    iteration = 0

    while cost - V > E and iteration < 10:
        print(f"Split cluster globale {cluster_2_split}")

        # 🔧 CORREZIONE: trova l'indice locale del cluster da splittare
        cluster_ids = sorted(map_cluster_classi.keys())
        if cluster_2_split in cluster_ids:
            cluster_index = cluster_ids.index(cluster_2_split)
        else:
            print(f"⚠️ Cluster ID {cluster_2_split} non trovato in map_cluster_classi, uso 0 di default.")
            cluster_index = 0

        if cluster_index >= membership_matrix.shape[1]:
            print(f"⚠️ Indice cluster_to_split {cluster_index} fuori range per U con shape {membership_matrix.shape}")
            break

        U, v = split_cluster(data=train, context=membership_matrix[:, cluster_index], C=n_clusters)

        if U is not None and v is not None:
            train, membership_matrix, prototypes, map_cluster_classi = replace_cluster(
                data=train, U=membership_matrix, v=prototypes,
                cluster=cluster_2_split, new_U=U, new_v=v,
                map_cluster_class=map_cluster_classi
            )
            n_clusters = len(map_cluster_classi)
            V = cost
            cost, cluster_2_split = select_cluster(X=train, U=membership_matrix, v=prototypes,
                                                   crit=reconstruction_error, path_result=output, maximize=True)
            vMaxList.append((cost, "Split", len(prototypes)))
            iteration += 1
        else:
            iteration = 10
            V = cost

    if output is not None:
        pd.DataFrame(membership_matrix).to_csv(os.path.join(output, f"membership_matrix_chunk{num_chunk}.csv"), index=False, header=False)
        pd.DataFrame(prototypes).to_csv(os.path.join(output, f"prototypes_chunk{num_chunk}.csv"), index=False, header=False)

    M = buildMemory(prototypes)
    return membership_matrix, M[0], M, V, map_cluster_classi, vMaxList

# DISSFCM senza split iniziale

def dissfcm_ns(X, fuzziness_coefficient, max_iter=100, alpha=1.0,
               stop_condition=('obj_delta', 0.001), distance='euclidean', M=None, map_cluster_classi=None,
               cluster_per_class=1, V=0, E=0, output=None, num_chunk=0, vMaxList=None):

    os.makedirs(output, exist_ok=True)
    if vMaxList is None:
        vMaxList = []

    train, map_cluster_classi = mapClusters(X, fuzziness_coefficient, map_cluster_classi, cluster_per_class)
    n_clusters = len(map_cluster_classi)

    b, F = get_b_F(X=train, n_clusters=n_clusters)
    membership_matrix, prototypes = issfcm(X=train, number_of_clusters=n_clusters,
                                           fuzziness_coefficient=fuzziness_coefficient, b=b, F=F, alpha=alpha,
                                           max_iter=max_iter, stop_condition=stop_condition, distance=distance,
                                           M=M, output=output)

    cost, cluster_2_split = select_cluster(X=train, U=membership_matrix,
                                           v=prototypes, crit=reconstruction_error,
                                           path_result=output, maximize=True)
    vMaxList.append((cost, "", len(prototypes)))
    iteration = 0

    while cost - V > E and iteration < 10:
        print(f"Split cluster globale {cluster_2_split}")

        # 🔧 CORREZIONE anche qui: mappa indice globale → locale
        cluster_ids = sorted(map_cluster_classi.keys())
        if cluster_2_split in cluster_ids:
            cluster_index = cluster_ids.index(cluster_2_split)
        else:
            print(f"⚠️ Cluster ID {cluster_2_split} non trovato in map_cluster_classi, uso 0 di default.")
            cluster_index = 0

        if cluster_index >= membership_matrix.shape[1]:
            print(f"⚠️ Indice cluster_to_split {cluster_index} fuori range per U con shape {membership_matrix.shape}")
            break

        U, v = split_cluster(data=train, context=membership_matrix[:, cluster_index], C=n_clusters)

        if U is not None and v is not None:
            train, membership_matrix, prototypes, map_cluster_classi = replace_cluster(
                data=train, U=membership_matrix, v=prototypes,
                cluster=cluster_2_split, new_U=U, new_v=v,
                map_cluster_class=map_cluster_classi
            )
            n_clusters = len(map_cluster_classi)
            V = cost
            cost, cluster_2_split = select_cluster(X=train, U=membership_matrix, v=prototypes,
                                                   crit=reconstruction_error, path_result=output, maximize=True)
            vMaxList.append((cost, "Split", len(prototypes)))
            iteration += 1
        else:
            iteration = 10

    if output is not None:
        pd.DataFrame(membership_matrix).to_csv(os.path.join(output, f"membership_matrix_chunk{num_chunk}.csv"), index=False, header=False)
        pd.DataFrame(prototypes).to_csv(os.path.join(output, f"prototypes_chunk{num_chunk}.csv"), index=False, header=False)

    M = buildMemory(prototypes)
    return membership_matrix, M[0], M, V, map_cluster_classi, vMaxList
