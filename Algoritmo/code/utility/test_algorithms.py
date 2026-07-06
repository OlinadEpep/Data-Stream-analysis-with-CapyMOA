import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from CFCM.cluster_splitting import select_cluster, reconstruction_error, split_cluster, replace_cluster
from DISSFCM.dissfcm import dissfcm
from FCM.fcm import fcm
from ISSFCM.issfcm import issfcm, buildMemory
from SSFCM.ssfcm import get_b_F, ssfcm
from utility.test_utilities import test_metrics, plotVMax, plot_clusters_new, plot_clusters, test, plotSimple, \
    get_class_distribution, get_train_and_test, test_with_map
from utility.utilities import distinct, create_clustering, fixPrototypes, get_dataset_partially_labelled, \
    evaluation_alpha


# l'inizializzazione dei centroidi in FCM avviene in maniera casuale e non prendendo un campione per ogni classe
# pertanto la corrispondenza indice 0 (della matrice di membership e dei prototipi), classe 0 non è assicurata questa
# funzione controlla che le etichette dei prototipi siano in ordine crescente ed eventualmente li sistema in maniera
# ordinata
def hard_fix(prototypes, membership_matrix):
    labels = list(prototypes[:, -1])
    if not all(labels[i] <= labels[i + 1] for i in range(len(labels) - 1)):
        new_prototypes = np.copy(prototypes)
        new_membership_matrix = np.copy(membership_matrix)
        for e, p in enumerate(prototypes):
            index = int(round(p[-1]))
            p[-1] = index
            new_prototypes[index, :] = np.reshape(p, newshape=(1, len(p)))
            new_membership_matrix[:, index] = membership_matrix[:, e]
        return new_prototypes, new_membership_matrix
    return fixPrototypes(prototypes), membership_matrix


def executeDISSFCM(data, shuffle=False, batch_size=100, fuzziness_coefficient=2, max_iter=100,
                   cluster_per_class=1, stop_condition=('obj_delta', 0.001), distance='euclidean', alpha=1.0,
                   output=None):
    M = None
    metrics_matrix = {}
    if shuffle:
        np.random.shuffle(data)
    index = 0
    num_chunk = 0
    V = 0
    map_cluster_classi = {}
    vMaxList = []
    while index < np.shape(data)[0]:
        chunk = data[index:index + batch_size, ]
        classDistribution = get_class_distribution(chunk)
        train_data, test_data = train_test_split(chunk, train_size=0.7)
        num_chunk += 1
        print("Chunk #" + str(num_chunk))

        membership_matrix, prototypes, M, V, map_cluster_classi, vMaxList = dissfcm(X=train_data,
                                                                                    fuzziness_coefficient=fuzziness_coefficient,
                                                                                    map_cluster_classi=map_cluster_classi,
                                                                                    max_iter=max_iter, alpha=alpha,
                                                                                    stop_condition=stop_condition,
                                                                                    distance=distance, M=M,
                                                                                    output=output,
                                                                                    cluster_per_class=cluster_per_class,
                                                                                    V=V,
                                                                                    num_chunk=num_chunk,
                                                                                    vMaxList=vMaxList)
        metrics_l = test_metrics(test_data, prototypes, map_cluster_classi)
        _, prediction, _ = test(test_data, prototypes)
        try:
            silhouette = metrics.silhouette_score(test_data[:, :-1], prediction, metric='euclidean')
            davies = metrics.davies_bouldin_score(test_data[:, :-1], prediction)
        except:
            silhouette = "N/A"
            davies = "N/A"

        print("Chunk #" + str(num_chunk) + " Accuracy: " + str(metrics_l[0]))
        metrics_l = metrics_l + [silhouette, davies,classDistribution]
        metrics_matrix[num_chunk] = metrics_l
        index += batch_size
    plotVMax(vMaxList, output)
    return metrics_matrix


# Permette l'esecuzione di ISSFCM a cui è aggiunto lo split dei cluster
# - data: i dati su cui eseguire l'algoritmo'
# - shuffle: mescola i dati prima di eseguire l'algoritmo, default True
# - batch_size: grandezza del chunk, default 100
# -output: path in cui salvare i plot prodotti e i file contenenti matrici di membership prodotte, prototipi e rec errors
#         default None, non viene salvato nessun output
#
# return:
#     matrice di metriche in cui ogni riga riporta accuracy, precision,recall,fscore e support
def executeISSFCM_with_split(data, shuffle=False, batch_size=100, fuzziness_coefficient=2, max_iter=100,
                             stop_condition=('obj_delta', 0.001), distance='euclidean', alpha=1.0, output=None):
    M = None
    metrics_matrix = {}
    if shuffle:
        np.random.shuffle(data)
    index = 0
    num_chunk = 0
    labs = distinct(data[:, -1])
    map_cluster_classi = {}
    for l in labs:
        map_cluster_classi[l] = l
    V = 0
    while index < np.shape(data)[0]:
        chunk = data[index:index + batch_size, ]
        train_data, test_data = train_test_split(chunk, train_size=0.7)
        num_chunk += 1
        print("Chunk #" + str(num_chunk))

        n_clusters = len(map_cluster_classi)
        b, F = get_b_F(X=train_data, n_clusters=n_clusters)
        membership_matrix, prototypes = issfcm(X=train_data, number_of_clusters=n_clusters,
                                               fuzziness_coefficient=fuzziness_coefficient, b=b, F=F, alpha=alpha,
                                               max_iter=max_iter, stop_condition=stop_condition, distance=distance,
                                               M=M, output=output)
        if output is not None:
            data_clustered = create_clustering(train_data, membership_matrix)
            t_c = "ISSFCM inizio Chunk " + str(num_chunk)
            plot_clusters_new(n=0, centroids=prototypes, clusters=data_clustered, map=map_cluster_classi,
                              path_result=output, namefile_result="ISSFCM_Chunk" + str(num_chunk) + "_inizio",
                              title_image=t_c)
        cost, cluster_2_split, idx_2_split = select_cluster(X=train_data, U=membership_matrix,
                                                            v=prototypes, crit=reconstruction_error,
                                                            path_result=output + "/rec_error.txt",
                                                            maximize=True)

        iteration = 0
        while cost > V and iteration < 10:
            print("Split cluster %s" % str(cluster_2_split))

            U, v = split_cluster(data=train_data, context=membership_matrix[:, idx_2_split],
                                 C=len(map_cluster_classi.keys()))

            train, membership_matrix, prototypes, map_cluster_classi = replace_cluster(data=train_data,
                                                                                       U=membership_matrix,
                                                                                       v=prototypes,
                                                                                       cluster=idx_2_split,
                                                                                       new_U=U,
                                                                                       new_v=v,
                                                                                       map_cluster_class=map_cluster_classi)
            if output is not None:
                data_clustered = create_clustering(X=train, U=membership_matrix)
                t_c = "ISSFCM Chunk" + str(num_chunk) + " Split #" + str(iteration + 1)
                plot_clusters_new(n=iteration + 1, centroids=prototypes, clusters=data_clustered,
                                  map=map_cluster_classi,
                                  path_result=output,
                                  namefile_result=t_c, title_image=t_c)

            V = cost
            cost, cluster_2_split, idx_2_split = select_cluster(X=train, U=membership_matrix, v=prototypes,
                                                                crit=reconstruction_error,
                                                                path_result=output + "/rec_error.txt",
                                                                maximize=True)
            iteration += 1

        if output is not None:
            mm = pd.DataFrame(membership_matrix)
            c = pd.DataFrame(prototypes)
            mm.to_csv(output + "/membership_matrix_chunk" + str(num_chunk) + ".csv", index=False, header=False)
            c.to_csv(output + "/prototypes_chunk" + str(num_chunk) + ".csv", index=False, header=False)
        M = buildMemory(prototypes)
        metrics = test_metrics(test_data, prototypes, map_cluster_classi)
        print("Chunk #" + str(num_chunk) + " Accuracy: " + str(metrics[0]))
        metrics_matrix[num_chunk] = metrics

        index += batch_size
    return metrics_matrix


def executeISSFCM(data, shuffle=False, batch_size=100, fuzziness_coefficient=2, max_iter=100,
                  stop_condition=('obj_delta', 0.001), distance='euclidean', alpha=1.0, output=None):
    M = None
    metrics_matrix = {}
    if shuffle:
        np.random.shuffle(data)
    index = 0
    num_chunk = 0

    while index < np.shape(data)[0]:
        chunk = data[index:index + batch_size, ]
        train_data, test_data = train_test_split(chunk, train_size=0.7)
        num_chunk += 1
        print("Chunk #" + str(num_chunk))

        n_clusters = len(M[0]) if M is not None else 2
        b, F = get_b_F(X=train_data, n_clusters=n_clusters)
        membership_matrix, prototypes = issfcm(X=train_data, number_of_clusters=n_clusters,
                                               fuzziness_coefficient=fuzziness_coefficient, b=b, F=F, alpha=alpha,
                                               max_iter=max_iter, stop_condition=stop_condition, distance=distance,
                                               M=M, output=output)
        if output is not None:
            data_clustered = create_clustering(train_data, membership_matrix)
            t_c = "ISSFCM inizio Chunk " + str(num_chunk)
            plot_clusters(n=0, centroids=prototypes, clusters=data_clustered,
                          path_result=output, namefile_result="ISSFCM_Chunk" + str(num_chunk) + "_inizio",
                          title_image=t_c)

            mm = pd.DataFrame(membership_matrix)
            c = pd.DataFrame(prototypes)
            mm.to_csv(output + "/membership_matrix_chunk" + str(num_chunk) + ".csv", index=False, header=False)
            c.to_csv(output + "/prototypes_chunk" + str(num_chunk) + ".csv", index=False, header=False)
        M = buildMemory(prototypes)
        labels, prediction, accuracy = test(test_data, prototypes)
        prfs = precision_recall_fscore_support(y_true=labels, y_pred=prediction)
        try:
            silhouette = metrics.silhouette_score(test_data[:, :-1], prediction, metric='euclidean')
            davies = metrics.davies_bouldin_score(test_data[:, :-1], prediction)
        except:
            silhouette = "N/A"
            davies = "N/A"
        metrics_l = [accuracy] + list(prfs) + [silhouette, davies]
        print("Chunk #" + str(num_chunk) + " Accuracy: " + str(accuracy))
        metrics_matrix[num_chunk] = metrics_l

        index += batch_size
    return metrics_matrix


# Permette l'esecuzione di FCM
# - data: i dati su cui eseguire l'algoritmo
# - n_cluster: numero di cluster da individuare
# -dataset_results: path in cui salvare i plot prodotti e i file contenenti matrici di membership prodotte, prototipi
#         default None, non viene salvato nessun output
#
# return:
#     metrics:  accuracy, precision,recall,fscore e support
def executeFCM(data, n_clusters, dataset_results, fuzziness_coefficient=2):
    train_data, test_data = train_test_split(data, train_size=0.7)
    membership_matrix, prototypes = fcm(X=train_data,
                                        number_of_clusters=n_clusters,
                                        fuzziness_coefficient=fuzziness_coefficient)
    fixPrototypes(prototypes)
    if dataset_results is not None:
        t_c = "FCM"
        plotSimple(train_data, membership_matrix, t_c, dataset_results)
        mm = pd.DataFrame(membership_matrix)
        c = pd.DataFrame(prototypes)
        mm.to_csv(dataset_results + "/membership_matrix.csv", index=False, header=False)
        c.to_csv(dataset_results + "/prototypes.csv", index=False, header=False)
    labels, prediction, accuracy = test(test_data, prototypes)
    prfs = precision_recall_fscore_support(y_true=labels, y_pred=prediction)
    try:
        silhouette = metrics.silhouette_score(test_data[:, :-1], prediction, metric='euclidean')

    except:
        silhouette = "N/A"
    # silhouette = metrics.silhouette_score(labels,prediction,metric='euclidean')
    metrics_ = [accuracy] + list(prfs) + [silhouette]
    print("Accuracy:" + str(accuracy))
    return metrics_


def executeFCM_with_split(data, n_clusters, dataset_results, fuzziness_coefficient=2):
    train_data, test_data = train_test_split(data, train_size=0.7)
    V = 0
    membership_matrix, prototypes = fcm(X=train_data,
                                        number_of_clusters=n_clusters,
                                        fuzziness_coefficient=fuzziness_coefficient)
    prototypes, membership_matrix = hard_fix(prototypes, membership_matrix)
    labs = prototypes[:, -1]
    map_cluster_classi = {}
    for l in labs:
        map_cluster_classi[l] = l
    if dataset_results is not None:
        data_clustered = create_clustering(X=train_data, U=membership_matrix)
        t_c = "FCM inizio"
        plot_clusters_new(n=0, centroids=prototypes, clusters=data_clustered, map=map_cluster_classi,
                          path_result=dataset_results, namefile_result="FCM_inizio",
                          title_image=t_c)
    if dataset_results is None: dataset_results = ""
    cost, cluster_2_split, idx_2_split = select_cluster(X=train_data, U=membership_matrix,
                                                        v=prototypes, crit=reconstruction_error,
                                                        path_result=dataset_results + "/rec_error.txt",
                                                        maximize=True)

    iteration = 0
    while cost > V and iteration < 10:
        print("Split cluster %s" % str(cluster_2_split))
        U, v = split_cluster(data=train_data, context=membership_matrix[:, idx_2_split],
                             C=len(map_cluster_classi.keys()))
        train, membership_matrix, prototypes, map_cluster_classi = replace_cluster(data=train_data,
                                                                                   U=membership_matrix,
                                                                                   v=prototypes,
                                                                                   cluster=idx_2_split,
                                                                                   new_U=U,
                                                                                   new_v=v,
                                                                                   map_cluster_class=map_cluster_classi)
        if dataset_results is not None:
            data_clustered = create_clustering(X=train, U=membership_matrix)
            t_c = "FCM Split #" + str(iteration + 1)
            plot_clusters_new(n=iteration + 1, centroids=prototypes, clusters=data_clustered, map=map_cluster_classi,
                              path_result=dataset_results,
                              namefile_result=t_c, title_image=t_c)
        V = cost
        cost, cluster_2_split, idx_2_split = select_cluster(X=train, U=membership_matrix, v=prototypes,
                                                            crit=reconstruction_error,
                                                            path_result=dataset_results + "/rec_error.txt",
                                                            maximize=True)
        iteration += 1
    if dataset_results is not None:
        file = open(dataset_results + "/Log.txt", "a")
        file.write("Membership Matrix\n")
        file = open(dataset_results + "/Log.txt", "ab")
        np.savetxt(file, X=membership_matrix, fmt='%.3f', delimiter=',')
        file = open(dataset_results + "/Log.txt", "a")
        file.write("\nPrototypes\n")
        file = open(dataset_results + "/Log.txt", "ab")
        np.savetxt(file, X=prototypes, fmt='%.3f', delimiter=',')
    metrics = test_metrics(test_data, prototypes, map_cluster_classi)
    print("Accuracy:" + str(metrics[0]))
    return metrics


# Permette l'esecuzione di FCM
# - data: i dati su cui eseguire l'algoritmo
# - n_cluster: numero di cluster da individuare
# -dataset_results: path in cui salvare i plot prodotti e i file contenenti matrici di membership prodotte, prototipi
#         default None, non viene salvato nessun output
#
# return:
#     metrics:  accuracy, precision,recall,fscore e support
def executeSSFCM(data, n_clusters, dataset_results, fuzziness_coefficient=2, alpha=1.0, max_iter=100,
                 stop_condition=('obj_delta', 0.001), distance='euclidean'):
    train_data, test_data = train_test_split(data, train_size=0.7)
    b, F = get_b_F(X=train_data, n_clusters=n_clusters)
    membership_matrix, prototypes = ssfcm(X=train_data, number_of_clusters=n_clusters,
                                          fuzziness_coefficient=fuzziness_coefficient, b=b, F=F, alpha=alpha,
                                          max_iter=max_iter, stop_condition=stop_condition, distance=distance)

    if dataset_results is not None:
        t_c = "SSFCM"
        plotSimple(train_data, membership_matrix, t_c, dataset_results)

        mm = pd.DataFrame(membership_matrix)
        c = pd.DataFrame(prototypes)
        mm.to_csv(dataset_results + "/membership_matrix.csv", index=False, header=False)
        c.to_csv(dataset_results + "/prototypes.csv", index=False, header=False)
    labels, prediction, accuracy = test(test_data, prototypes)
    prfs = precision_recall_fscore_support(y_true=labels, y_pred=prediction)
    try:
        silhouette = metrics.silhouette_score(test_data[:, :-1], prediction, metric='euclidean')

    except:
        silhouette = "N/A"
    # silhouette = metrics.silhouette_score(labels,prediction)
    metrics_ = [accuracy] + list(prfs) +[silhouette]
    print("Accuracy:" + str(accuracy))
    return metrics_


def executeSSFCM_with_split(data, n_clusters, dataset_results, fuzziness_coefficient=2, alpha=1.0, max_iter=100,
                            stop_condition=('obj_delta', 0.001), distance='euclidean'):
    train_data, test_data = train_test_split(data, train_size=0.7)
    V = 0

    b, F = get_b_F(X=train_data, n_clusters=n_clusters)
    membership_matrix, prototypes = ssfcm(X=train_data, number_of_clusters=n_clusters,
                                          fuzziness_coefficient=fuzziness_coefficient, b=b, F=F, alpha=alpha,
                                          max_iter=max_iter, stop_condition=stop_condition, distance=distance)
    labs = distinct(prototypes[:, -1])
    map_cluster_classi = {}
    for l in labs:
        map_cluster_classi[l] = l

    if dataset_results is not None:
        data_clustered = create_clustering(X=train_data, U=membership_matrix)
        t_c = "SSFCM inizio"
        plot_clusters_new(n=0, centroids=prototypes, clusters=data_clustered, map=map_cluster_classi,
                          path_result=dataset_results, namefile_result="SSFCM_inizio",
                          title_image=t_c)
    if dataset_results is None: dataset_results = ""
    cost, cluster_2_split, idx_2_split = select_cluster(X=train_data, U=membership_matrix,
                                                        v=prototypes, crit=reconstruction_error,
                                                        path_result=dataset_results + "/rec_error.txt",
                                                        maximize=True)

    iteration = 0
    while cost > V and iteration < 10:
        print("Split cluster %s" % str(cluster_2_split))
        U, v = split_cluster(data=train_data, context=membership_matrix[:, idx_2_split],
                             C=len(map_cluster_classi.keys()))
        train, membership_matrix, prototypes, map_cluster_classi = replace_cluster(data=train_data,
                                                                                   U=membership_matrix,
                                                                                   v=prototypes,
                                                                                   cluster=idx_2_split,
                                                                                   new_U=U,
                                                                                   new_v=v,
                                                                                   map_cluster_class=map_cluster_classi)
        if dataset_results is not None:
            data_clustered = create_clustering(X=train, U=membership_matrix)
            t_c = "SSFCM Split #" + str(iteration + 1)
            plot_clusters_new(n=iteration + 1, centroids=prototypes, clusters=data_clustered, map=map_cluster_classi,
                              path_result=dataset_results,
                              namefile_result=t_c, title_image=t_c)
        V = cost
        cost, cluster_2_split, idx_2_split = select_cluster(X=train, U=membership_matrix, v=prototypes,
                                                            crit=reconstruction_error,
                                                            path_result=dataset_results + "/rec_error.txt",
                                                            maximize=True)
        iteration += 1
    if dataset_results is not None:
        mm = pd.DataFrame(membership_matrix)
        c = pd.DataFrame(prototypes)
        mm.to_csv(dataset_results + "/membership_matrix_chunk.csv", index=False, header=False)
        c.to_csv(dataset_results + "/prototypes_chunk.csv", index=False, header=False)
    metrics = test_metrics(test_data, prototypes, map_cluster_classi)
    print("Accuracy: " + str(metrics[0]))
    return metrics


def execute_dissfcm_sperimentazione(repetition, chunks, p, clusters_per_class, execution_info_s,
                                    classDistributionTotale, name, output):
    fuzziness_coefficient = 2
    i = 1
    while i <= repetition:
        M = None
        V = 0
        acc = []
        silhouetteL = []
        daviesL = []
        precisionL = []
        recallL = []
        f_scoreL = []
        vMaxList = []
        map_cluster_classi = {}
        count = 1
        for chunk in chunks:
            classDistribution = get_class_distribution(chunk)
            train_data, test_data = get_train_and_test(chunk, test_size=20)
            if p < 100:
                train_data = get_dataset_partially_labelled(train_data, p)
            samples = np.shape(train_data)[0]
            n_labeled = len(list(filter(lambda x: x != -1, list(train_data[:, -1]))))
            alpha = evaluation_alpha(p, samples, n_labeled)
            test_data = test_data[:, 1:]
            train_data = train_data[:, 1:]
            membership_matrix, prototypes, M, V, map_cluster_classi, vMaxList = dissfcm(X=train_data,
                                                                                        fuzziness_coefficient=fuzziness_coefficient,
                                                                                        map_cluster_classi=map_cluster_classi,
                                                                                        M=M,
                                                                                        cluster_per_class=clusters_per_class,
                                                                                        V=V,
                                                                                        output=output + "/p_labelled" + str(
                                                                                            p) + "/" + str(i),
                                                                                        num_chunk=count,
                                                                                        alpha=alpha,
                                                                                        vMaxList=vMaxList)
            true, predicted, metrics_T = test_with_map(test_data, prototypes, map_cluster_classi)
            _, pred_sd, _ = test(test_data, prototypes)
            try:
                silhouette = metrics.silhouette_score(test_data[:, :-1], pred_sd, metric='euclidean')
                davies = metrics.davies_bouldin_score(test_data[:, :-1], pred_sd)
            except:
                silhouette = "N/A"
                davies = "N/A"

            acc.append(metrics_T[0])
            silhouetteL.append(silhouette)
            daviesL.append(davies)
            precisionL.append(metrics_T[1])
            recallL.append(metrics_T[2])
            f_scoreL.append(metrics_T[3])
            execution_info_s["Dataset_name"].append(str(name))
            execution_info_s["Repetition"].append(str(i))
            execution_info_s["Chunk"].append(count)
            execution_info_s["p_entry_labelled"].append(str(p))
            execution_info_s["Class_Distribution"].append(str(classDistribution))
            execution_info_s["Accuracy"].append(str(metrics_T[0]))
            execution_info_s["Silhouette"].append(str(silhouette))
            execution_info_s["Recall"].append(metrics_T[2])
            execution_info_s["Precision"].append(metrics_T[1])
            execution_info_s["F_Score"].append(metrics_T[3])
            execution_info_s["Davies"].append(str(davies))

            # print(metrics_T[0])
            count += 1

        t_acc = np.mean(acc)
        print("Accuracy:%s" % t_acc)
        davies = np.mean(list(filter(lambda a: a != "N/A", daviesL)))
        silhouette = np.mean(list(filter(lambda a: a != "N/A", silhouetteL)))
        precision = np.mean(precisionL, axis=0)
        recall = np.mean(recallL, axis=0)
        f_score = np.mean(f_scoreL, axis=0)

        execution_info_s["Dataset_name"].append(str(name))
        execution_info_s["Repetition"].append(str(i))
        execution_info_s["Chunk"].append("Totale")
        execution_info_s["p_entry_labelled"].append(str(p))
        execution_info_s["Class_Distribution"].append(str(classDistributionTotale))
        execution_info_s["Accuracy"].append(str(t_acc))
        execution_info_s["Recall"].append(recall)
        execution_info_s["Precision"].append(precision)
        execution_info_s["F_Score"].append(f_score)
        execution_info_s["Silhouette"].append(str(silhouette))
        execution_info_s["Davies"].append(str(davies))
        plotVMax(vMaxList, output, "p" + str(p) + "rip" + str(i))

        i += 1
