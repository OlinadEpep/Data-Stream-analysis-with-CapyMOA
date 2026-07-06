import numpy as np
from scipy.spatial import distance_matrix
from skmultiflow.core import BaseSKMObject, ClassifierMixin
from DISSFCM.dissfcm import dissfcm


# classe che implementa DISSFCM per multiflow
class DISSFCM(BaseSKMObject, ClassifierMixin):

    # metodo per inizializzare dissfcm, in questo momento vanno indicati tutti i parametri richiesti dal metodo
    # dissfcm ad esclusione dei dati da utilizzare che verranno passati seguendo il workflow di multiflow l'unico
    # parametro escluso è vMaxlist che non è utilizzato in quanto multiflow non permette il controllo sui plot da
    # eseguire a termine esecuzione
    def __init__(self,
                 alpha,
                 fuzziness_coefficient=2,
                 max_iter=100,
                 stop_condition=('obj_delta', 0.001),
                 distance='euclidean',
                 output=None,
                 V=0,
                 E=0,
                 clusters_per_class=1
                 ):
        super().__init__()

        self.map_cluster_classi = {}
        self.alpha = alpha
        self.fuzziness_coefficient = fuzziness_coefficient
        self.max_iter = max_iter
        self.stop_condition = stop_condition
        self.distance = distance
        self.M = None
        self.prototypes = []
        self.output = output
        self.V = V
        self.E=E
        self.clusters_per_class = clusters_per_class
        self.num_chunk = 1
        return

    # esecuzione di dissfmc
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        data = np.concatenate((X, np.reshape(y, (y.shape[0], 1))), axis=1)
        membership_matrix, self.prototypes, self.M, self.V, self.map_cluster_classi, _ = dissfcm(X=data,
                                                                                                 fuzziness_coefficient=self.fuzziness_coefficient,
                                                                                                 map_cluster_classi=self.map_cluster_classi,
                                                                                                 max_iter=self.max_iter,
                                                                                                 stop_condition=self.stop_condition,
                                                                                                 distance=self.distance,
                                                                                                 alpha=self.alpha,
                                                                                                 M=self.M,
                                                                                                 output=self.output,
                                                                                                 cluster_per_class=self.clusters_per_class,
                                                                                                 V=self.V,
                                                                                                 E=self.E,
                                                                                                 num_chunk=self.num_chunk)
        self.num_chunk = self.num_chunk + 1
        return self

    # predict classes for the passed data
    def predict(self, X=None):
        distanceMatrix = self.predict_proba(X)
        samples = np.shape(X)[0]
        prediction = []
        a = np.argmin(distanceMatrix, axis=1)
        for i in range(samples):
            prediction.append(self.map_cluster_classi[round(a[i])])
        return prediction

    # produce la distance matrix tra i dati passati e i prototipi prodotti al passso precedente
    def predict_proba(self, X=None):
        return distance_matrix(X, self.prototypes[:, :-1])
