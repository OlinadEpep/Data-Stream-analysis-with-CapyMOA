import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix


def distinct(lista, ignore=-1):
    distinct_lista = list(set(lista))
    if ignore is not None:
        try:
            distinct_lista.remove(ignore)
        except ValueError:
            pass
    return distinct_lista


def evaluation_alpha(percentage, cardinality_dataset, n_labelled):
    if percentage == 0:
        alpha = 0.0
    else:
        alpha = cardinality_dataset / n_labelled
    return alpha


# Crea una versione parzialmente etichettata del dataset e salva il risultato su un file
# percentage è la percentuale di dati etichettati
def create_dataset_partially_labelled(dataset_path, percentage, random_seed, output):
    np.random.seed(random_seed)
    dataframe = pd.read_csv(dataset_path, delimiter=',', header=None)
    dataset = dataframe.values
    data = dataset[:, :]
    entry = []
    cont_labelled = 0
    cont_unlabelled = 0
    for i in range(np.shape(dataset)[0]):
        if percentage == 100:
            item = data[i:i + 1, :]
            entry.append(item[0])
            cont_labelled += 1
        else:
            r = np.random.binomial(n=1, p=percentage / 100, size=1)
            if r > (percentage / 100):
                item = data[i:i + 1, :]
                entry.append(item[0])
                cont_labelled += 1
            else:
                unlabelled = np.array(data[i:i + 1, 0:-1])
                unlabelled = np.append(unlabelled, -1)
                entry.append(unlabelled)
                cont_unlabelled += 1

    alpha = evaluation_alpha(percentage, np.shape(dataset)[0], cont_labelled)

    print("Etichettati: ", cont_labelled, "\t\tNon etichettati: ", cont_unlabelled, "\nPercentage:", str(percentage),
          "\t\tAlpha:", str(alpha))
    dataset_partial_labelled = pd.DataFrame(entry)
    dataset_partial_labelled.to_csv(output, index=None, header=None)
    return alpha, cont_labelled


# Crea una versione parzialmente etichettata del dataset e restituisce direttamente il dataset così modificato
# assicurando la stessa percentuale di "disetichettatura" per ogni classe
# percentage è il numero di dati etichettati
def get_dataset_partially_labelled(data, percentage):
    classes = list(set(data[:, -1]))
    new_matrix = np.zeros(shape=(0, np.shape(data)[1]))
    # print("Samples to unlabel: %s"%str(round((np.shape(data)[0] * percentage) / 100)))
    for c in classes:
        slice_data = data[data[:, -1] == c]
        unlabeldataset(slice_data, 100 - percentage)
        new_matrix = np.append(new_matrix, slice_data, axis=0)
    np.random.shuffle(new_matrix)
    # print("Samples unlabeled: %s"%str(np.shape(new_matrix[new_matrix[:,-1]==-1])[0]))
    return new_matrix


# funzione per rimuovere etichetta a dati classificati
# data: i dati su cui si vuole lavorare
# percentage: la percentuale di dati a cui rimuovere l'etichetta
def unlabeldataset(data, percentage):
    to_unlabel = round((np.shape(data)[0] * percentage) / 100)
    if to_unlabel == np.shape(data)[0]:
        data[:, -1] = -1
    elif to_unlabel != 0:
        import random
        i_to_unlabel = random.sample(range(1, np.shape(data)[0]), to_unlabel)
        for i in i_to_unlabel:
            data[i, -1] = -1
    return data, to_unlabel


def fixPrototypes(prototypes):
    index = 0
    n = np.shape(prototypes)[0]
    for i in range(n):
        prototypes[index][-1] = round(prototypes[index][-1])
        index += 1
    return prototypes


# Calcola il numero di campioni entichettati
def entry_labeled(data):
    m = [row for row in data if row[-1] != -1]
    return len(m)


# clusterizza i dati utilizzando il valore maggiore della membership_matrix
def create_clustering(X, U):
    number_of_samples = np.shape(U)[0]
    number_of_clusters = np.shape(U)[1]
    number_of_features = np.shape(X)[1]

    clusters = {}
    for index_cluster in range(number_of_clusters):
        clusters[index_cluster] = np.zeros(shape=(0, number_of_features))

    for s in range(number_of_samples):
        index = np.where(U[s] == np.max(U[s]))[0][0]
        clusters[index] = np.append(clusters[index], np.array(X[s:s + 1, :]), axis=0)

    return clusters


# clusterizza i dati utilizzando la vicinanza ai prototipi
def create_clustering_prototypes(X, v):
    number_of_samples = np.shape(X)[0]
    number_of_clusters = np.shape(v)[0]
    number_of_features = np.shape(X)[1]

    clusters = {}
    for index_cluster in range(number_of_clusters):
        clusters[index_cluster] = np.zeros(shape=(0, number_of_features))

    distanceMatrix = distance_matrix(X[:, :-1], v[:, :-1])
    a = np.argmin(distanceMatrix, axis=1)
    for s in range(number_of_samples):
        clusters[a[s]] = np.concatenate(
            (clusters[a[s]], np.reshape(np.array(X[s, :]), newshape=(1, number_of_features))), axis=0)
    return clusters


# clustering per dati parzialmente etichettati (utilizzando il valore della membership matrix)
def create_clustering_partially_labelled(X, U, initial_label):
    number_of_samples = np.shape(U)[0]
    number_of_clusters = np.shape(U)[1]
    number_of_features = np.shape(X)[1]

    clusters = {}
    detailed_view = {"datapoint": [], "labelled": [], "class": [], "cluster": []}
    for index_cluster in range(number_of_clusters):
        clusters[index_cluster] = np.zeros(shape=(0, number_of_features))

    for s in range(number_of_samples):
        index = np.where(U[s] == np.max(U[s]))[0][0]
        clusters[index] = np.append(clusters[index], np.array(X[s:s + 1, :]), axis=0)
        detailed_view["datapoint"].append(X[s:s + 1, :-1])
        if X[s:s + 1, -1] == -1.0:
            detailed_view["labelled"].append(0)
        else:
            detailed_view["labelled"].append(1)
        detailed_view["class"].append(int(initial_label[s:s + 1]))

        detailed_view["cluster"].append(index + 1)
    return clusters, detailed_view
