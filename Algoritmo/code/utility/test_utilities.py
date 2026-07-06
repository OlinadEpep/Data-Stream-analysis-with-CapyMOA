from matplotlib import cm
from scipy.spatial import distance_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random


# rimuove i dati non etichettati (label=-1)
def removeUnlabeled(train, test_data):
    train = train[train[:, -1] != -1]
    test_data = test_data[test_data[:, -1] != -1]
    return train, test_data


# ottenere la distribuzione della classi nei dati
def get_class_distribution(data):
    labels = data[:, -1]
    classes = set(list(labels))
    distribution = []
    for c in classes:
        distribution.append((c, np.shape(data[data[:, -1] == c])[0]))
    return distribution


# divide i dati in train set e test set prendendo i campioni in maniera alternata
# es. campione 1 -> train, campione 2->test, capione 3->train,ecc
def getTrainAndTest(data):
    i = 0
    train = []
    test_data = []
    while i < np.shape(data)[0]:
        train.append(data[i, :])
        i += 1
        if i < np.shape(data)[0]:
            test_data.append(data[i, :])
            i += 1
    return np.asarray(train), np.asarray(test_data)


# divide i dati in train set e test set dividendoli sencondo il test size fornito come parametro
# preservando la distribuzione delle classi
def get_train_and_test(data, test_size):
    classes = list(set(data[:, -1]))
    test_data = []
    train_data = []
    for c in classes:
        slice_data = data[data[:, -1] == c]
        test_samples = round((np.shape(slice_data)[0] * test_size) / 100)
        test_indices = random.sample(range(0, np.shape(slice_data)[0]), test_samples)
        for id, sample in enumerate(slice_data):
            if id in test_indices:
                test_data.append(sample)
            else:
                train_data.append(sample)
    test_data = np.array(test_data)
    train_data = np.array(train_data)
    np.random.shuffle(test_data)
    np.random.shuffle(train_data)
    return train_data, test_data


# funzione per plottare dati di un dataset con 2 features (o con le prime due features)
def plot_raw_data(n, data, path_result, namefile_result, title_image):
    marker = 'o'
    size = 50
    alpha = 1
    color = 'y'
    f = plt.figure(n, figsize=(8, 8))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')
    plt.scatter(data[:, 0:1], data[:, 1:2], marker=marker, color=color, s=size, edgecolors='black', linewidths=1.5,
                alpha=alpha, label="Datapoints")
    plt.legend(bbox_to_anchor=(1.3, 0.5), loc="lower right", labelspacing=2)
    plt.tight_layout()
    plt.title(title_image)
    plt.savefig(path_result + "/" + namefile_result + ".png")
    return


# Permette di plottare i dati a tre dimensioni utilizzando i valori di appartenenza della membership matrix
# usata in caso di multiclass (3 classi)
def plotSimple3D(data, membership_matrix, fileName=None, path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    samples = np.shape(data)[0]
    for i in range(samples - 1):
        label = data[i, -1]
        if label == 1:
            ax.scatter(membership_matrix[i, 0], membership_matrix[i, 1], membership_matrix[i, 2], color='red')
        elif label == 2:
            ax.scatter(membership_matrix[i, 0], membership_matrix[i, 1], membership_matrix[i, 2], color='blue')
        elif label == 3:
            ax.scatter(membership_matrix[i, 0], membership_matrix[i, 1], membership_matrix[i, 2], color='green')
        else:
            ax.scatter(membership_matrix[i, 0], membership_matrix[i, 1], membership_matrix[i, 2], color='gray',
                       alpha=0.4)

    ax.set_xlabel("Class 1")
    ax.set_ylabel("Class 2")
    ax.set_zlabel("Class 3")

    class1 = mpatches.Patch(color='red', label='Class 1')
    class2 = mpatches.Patch(color='blue', label='Class 2')
    class3 = mpatches.Patch(color='green', label='Class 3')
    unlabeled = mpatches.Patch(color='gray', label='Unlabeled')

    plt.legend(handles=[class1, class2, class3, unlabeled])
    if path is not None and fileName is not None:
        plt.savefig(path + "/" + fileName + ".png")
    plt.clf()


# Permette di plottare i dati a utilizzando i valori di appartenenza della membership matrix
# usata in caso di calssificazione binaria
def plotSimple(data, membership_matrix, fileName=None, path=None):
    samples = np.shape(data)[0]
    for i in range(samples):
        label = data[i, -1]
        if label == 0:
            plt.scatter(membership_matrix[i, 0], membership_matrix[i, 1], c='red')
        elif label == 1:
            plt.scatter(membership_matrix[i, 0], membership_matrix[i, 1], c='blue')
        else:
            plt.scatter(membership_matrix[i, 0], membership_matrix[i, 1], c='gray', alpha=0.4)

    plt.xlabel("Class 1")
    plt.ylabel("Class 2")

    class1 = mpatches.Patch(color='red', label='Class 1')
    class2 = mpatches.Patch(color='blue', label='Class 2')
    unlabeled = mpatches.Patch(color='gray', label='Unlabeled')

    plt.legend(handles=[class1, class2, unlabeled])
    if path is not None and fileName is not None:
        plt.savefig(path + "/" + fileName + ".png")
    plt.clf()


# plot per dataset parzialmente etichettati
# nota F.G.: non  ho mai usato questa funzione,non saprei dire se funziona o meno
def plot_partially_clusters(n, centroids, clusters, path_result, namefile_result, title_image):
    marker = {"sample": 'o', "centroid": 'X'}
    size = {"sample": 80, "centroid": 250}
    alpha = {"sample": 0.5, "centroid": 1}
    colors = cm.rainbow(np.linspace(0, 1, num=centroids.shape[0]))

    ks = list(clusters.keys())
    n_samples_used = len(clusters[str(ks[0])])
    f = plt.figure(n, figsize=(12, 12))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')
    example = {}
    for x in range(np.shape(centroids)[0]):
        example[str(x)] = {}
    for k, v in example.items():
        labelled_right = []
        labelled_wrong = []
        unlabelled_right = []
        unlabelled_wrong = []
        v["labelled_r"] = labelled_right
        v["labelled_w"] = labelled_wrong
        v["unlabelled_r"] = unlabelled_right
        v["unlabelled_w"] = unlabelled_wrong
    for each_sample in range(n_samples_used):
        point = clusters[ks[0]][each_sample][0]
        is_labelled = clusters[ks[1]][each_sample]
        cls = clusters[ks[2]][each_sample]
        cluster_which_belongs = clusters[ks[3]][each_sample]
        if is_labelled == 0:
            if cls == cluster_which_belongs:
                example[str(cluster_which_belongs - 1)]["unlabelled_r"].append(point)
            else:
                example[str(cluster_which_belongs - 1)]["unlabelled_w"].append(point)
        else:
            if cls == cluster_which_belongs:
                example[str(cluster_which_belongs - 1)]["labelled_r"].append(point)
            else:
                example[str(cluster_which_belongs - 1)]["labelled_w"].append(point)
    a = 0
    lw = 1
    for k, v in example.items():
        number_of_cluster_examined = int(k)
        for sub_k, sub_v in v.items():
            ls = ''
            c = ''
            label = ""
            if sub_k == "labelled_r":
                lw = 1
                ls = '-'
                c = 'green'
                label = "labelled right - cluster " + str(int(k) + 1)
                a = 0.15
            elif sub_k == "labelled_w":
                lw = 1
                ls = '-'
                c = 'yellow'
                label = "labelled wrong - cluster " + str(int(k) + 1)
                a = 1
            elif sub_k == "unlabelled_r":
                ls = ':'
                lw = 3
                c = 'green'
                label = "unlabelled right - cluster " + str(int(k) + 1)
                a = 0.15
            elif sub_k == "unlabelled_w":
                lw = 3
                ls = ':'
                c = 'yellow'
                a = 1
                label = "unlabelled wrong - cluster " + str(int(k) + 1)
            else:
                label = "boh"
            if len(sub_v) > 0:
                plt.scatter(np.array(sub_v)[:, 0:1], np.array(sub_v)[:, 1:2], marker=marker["sample"],
                            edgecolors=c, linewidths=lw,
                            color=colors[number_of_cluster_examined],
                            s=size["sample"], alpha=a, linestyle=ls,
                            label=label)
    cont = 0
    for c, i in zip(centroids, range(np.shape(centroids)[0])):
        plt.scatter(c[0], c[1], marker=marker["centroid"], edgecolors='black', linewidths=2, color=colors[i],
                    s=size["centroid"], alpha=alpha["centroid"],
                    label="Centroid_" + str(cont + 1))

        cont += 1
    plt.legend(bbox_to_anchor=(1.3, 0.5), loc="lower right", labelspacing=2)
    plt.tight_layout()
    plt.title(title_image)
    plt.savefig(path_result + "/" + namefile_result + ".png")
    return


# plot dei dati in base ai cluster trovati
def plot_clusters(n, centroids, clusters, path_result, namefile_result, title_image):
    marker = {"sample": 'o', "centroid": 'X'}
    size = {"sample": 50, "centroid": 150}
    alpha = {"sample": 0.4, "centroid": 1}
    colors = cm.rainbow(np.linspace(0, 1, num=centroids.shape[0]))

    f = plt.figure(n, figsize=(8, 8))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')
    for (k, v), i in zip(clusters.items(), range(np.shape(centroids)[0])):
        plt.scatter(clusters[k][:, 0:1], clusters[k][:, 1:2], marker=marker["sample"], color=colors[i],
                    s=size["sample"],
                    alpha=alpha["sample"], label="Cluster_" + str(k + 1))

    cont = 0
    for c, i in zip(centroids, range(np.shape(centroids)[0])):
        plt.scatter(c[0], c[1], marker=marker["centroid"], edgecolors='black', linewidths='1', color=colors[i],
                    s=size["centroid"], alpha=alpha["centroid"],
                    label="Centroid_" + str(cont + 1))

        cont += 1

    plt.legend(bbox_to_anchor=(1.3, 0.5), loc="lower right", labelspacing=2)
    plt.tight_layout()
    plt.title(title_image)
    plt.savefig(path_result + "/" + namefile_result + ".png")
    plt.clf()
    # plt.show()
    return


# versione aggiornata di plot clusters che utilizza il dizionario map per la corrispondeza cluster-classi
def plot_clusters_new(n, centroids, clusters, map, path_result, namefile_result, title_image):
    class_colors = {0: 'b', 1: 'r', 2: 'g', 3: 'y'}
    colors = {0: 'c', 1: 'y', 2: 'violet', 3: 'orange', 4: 'lime', 5: 'aqua', 6: 'pink', 7: 'coral',
              8: 'olive', 9: 'aquamarine', 10: 'deepskyblue', 11: 'navy', 12: 'purple', 13: 'gray'}
    marker = {"sample": 'o', "centroid": 'X'}
    size = {"sample": 250, "centroid": 350}
    alpha = {"sample": 1, "centroid": 1}

    f = plt.figure(n, figsize=(18, 18))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')
    for (k, v), i in zip(clusters.items(), range(np.shape(centroids)[0])):
        if len(clusters[k])!=0:
            plt.scatter(clusters[k][:, 0:1], clusters[k][:, 1:2], marker=marker["sample"],
                        edgecolors=class_colors[int(map[k])], color=colors[i], s=size["sample"],
                        alpha=alpha["sample"], label="Cluster_" + str(k + 1) + "(" + str(len(v)) + ")")

    cont = 0
    for c, i in zip(centroids, range(np.shape(centroids)[0])):
        if len(clusters[i])!=0:
            plt.scatter(c[0], c[1], marker=marker["centroid"], edgecolors='black', linewidths=3, color=colors[i],
                        s=size["centroid"], alpha=alpha["centroid"],
                        label="Centroid_" + str(cont + 1))

        cont += 1

    plt.legend(loc='best',fontsize='xx-large',prop={'size': 20})
    plt.tight_layout()
    plt.title(title_image)
    print(path_result + "/" + namefile_result + ".png")
    plt.savefig(path_result + "/" + namefile_result + ".png") #errore in questa riga nel salvataggio della figura plottata.
    plt.close()
    return


# plot dei medoid
# nota F.G: in caso di dati numerosi, l'esecuzione è lentissima. Non avendo avuto necessità di
# utilizzarla, non l'ho sistemata
def plot_centroids_medoids(n, centroids, clusters, medoids, path_result, namefile_result, title_image):
    marker = {"sample": 'o', "centroid": 'X', "medoid": 'o'}
    size = {"sample": 50, "centroid": 150, "medoid": 150}
    alpha = {"sample": 0.4, "centroid": 1, "medoid": 1}
    colors = cm.rainbow(np.linspace(0, 1, num=centroids.shape[0]))

    f = plt.figure(n, figsize=(8, 8))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')
    for (k, v), i in zip(clusters.items(), range(np.shape(centroids)[0])):
        plt.scatter(clusters[k][:, 0:1], clusters[k][:, 1:2], marker=marker["sample"], color=colors[i],
                    s=size["sample"],
                    alpha=alpha["sample"], label="Cluster_" + str(k + 1))

    cont = 0
    for c, i in zip(centroids, range(np.shape(centroids)[0])):
        plt.scatter(c[0], c[1], marker=marker["centroid"], edgecolors='black', linewidths='1', color=colors[i],
                    s=size["centroid"], alpha=alpha["centroid"],
                    label="Centroid_" + str(cont + 1))

        cont += 1

    cont_1 = 0
    for m, j in zip(medoids, range(np.shape(medoids)[0])):
        plt.scatter(m[0], m[1], marker=marker["medoid"], edgecolors='black', linewidths='1', color=colors[j],
                    s=size["medoid"], label="Medoid_" + str(cont_1 + 1),
                    alpha=alpha["medoid"])
        cont_1 += 1

    plt.legend(bbox_to_anchor=(1.3, 0.5), loc="lower right", labelspacing=2)
    plt.tight_layout()
    plt.title(title_image)
    plt.savefig(path_result + "/" + namefile_result + ".png")
    return


def test_with_map(test_data, prototipi, mapping):
    labels = test_data[:, -1]
    distanceMatrix = distance_matrix(test_data[:, :-1], prototipi[:, :-1])
    samples = np.shape(test_data)[0]
    prediction = []
    acc_test = []
    acc_pred = []
    a = np.argmin(distanceMatrix, axis=1)
    for i in range(samples):
        prediction.append(mapping[a[i]])
        if labels[i] != -1:
            acc_test.append(labels[i])
            acc_pred.append(mapping[a[i]])

    metrics = [accuracy_score(y_true=np.asarray(acc_test),
                              y_pred=np.asarray(acc_pred))]  # calcolo l'accuracy solo sui dati etichettati
    t = precision_recall_fscore_support(y_true=np.asarray(acc_test), y_pred=np.asarray(acc_pred))
    for i in t:
        metrics.append(i)

    return labels.tolist(), prediction, metrics


def test(test_data, prototipi):
    labels = test_data[:, -1]
    classes = list(prototipi[:, -1])
    distanceMatrix = distance_matrix(test_data[:, :-1], prototipi[:, :-1])
    samples = np.shape(test_data)[0]
    prediction = []
    acc_test = []
    acc_pred = []
    a = np.argmin(distanceMatrix, axis=1)
    for i in range(samples):
        prediction.append(classes[a[i]])
        if labels[i] != -1:
            acc_test.append(labels[i])
            acc_pred.append(classes[a[i]])
    acc = accuracy_score(y_true=np.asarray(acc_test),
                         y_pred=np.asarray(acc_pred))  # calcolo l'accuracy solo sui dati etichettati
    return labels.tolist(), prediction, acc


# test_with_map ma con il calcolo delle metriche integrato
# metriche calcolate: accuracy, precision,recall,fscore,support
def test_metrics(test_data, prototipi, mapping):
    metrics = []
    labels = test_data[:, -1].tolist()
    distanceMatrix = distance_matrix(test_data[:, :-1], prototipi[:, :-1])
    samples = np.shape(test_data)[0]
    prediction = []
    a = np.argmin(distanceMatrix, axis=1)
    for i in range(samples):
        prediction.append(mapping[round(a[i])])
    metrics.append(accuracy_score(y_true=labels, y_pred=prediction))
    prfs = precision_recall_fscore_support(y_true=labels, y_pred=prediction)
    metrics = metrics + list(prfs)
    return metrics


# plot dei valori massimi del rec error
def plotVMax(vMaxList, output, extra=""):
    value = []
    label = []
    nclusters = []
    for v in vMaxList:
        value.append(v[0])
        label.append(v[1])
        nclusters.append(v[2])

    N = len(value)
    ind = np.arange(N)  # the x locations for the groups

    width = 0
    font = {'weight': 'bold',
            'size': 15}

    plt.matplotlib.rc('font', **font)
    fig, ax = plt.subplots()
    fig.set_size_inches(16.5, 8.5)
    # ax.scatter(ind, value)
    # add some text for labels, title and axes ticks
    ax.set_xticks(ind + width)
    ax.set_xticklabels(ind)

    for tick in ax.get_xticklabels():
        tick.set_rotation(-45)

    for index in range(N):
        if label[index] == "Split":
            ax.plot(ind[index], value[index], "or", color="blue",markersize=12)
        else:
            ax.plot(ind[index], value[index], "or", color="green",markersize=12)

        plt.annotate(nclusters[index], xy=(index+0.05, value[index]))

    ax.plot(ind, value, "-b")
    blue_circle = mpatches.Patch(color="blue", label="After split")
    green_circle = mpatches.Patch(color="green", label="New chunk")

    plt.legend(loc='lower left', prop={'size': 8}, bbox_to_anchor=(0.12, 1.02, 1, 0.2),
               handles=[green_circle, blue_circle], ncol=2)

    plt.savefig(output + "/VMaxPlot_" + str(extra), ext="png", close=False, verbose=True)
    # plt.show()
    plt.close()
