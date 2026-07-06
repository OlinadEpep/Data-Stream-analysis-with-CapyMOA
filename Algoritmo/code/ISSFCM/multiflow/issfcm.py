from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils.utils import *
from ISSFCM.issfcm import issfcm, buildMemory
from scipy.spatial import distance_matrix
from SSFCM.ssfcm import get_b_F


# classe che implementa ISSFCM per multiflow
# ClassifierMixin classe astratta da personalizzare
class ISSFCM(BaseSKMObject, ClassifierMixin):
    # metodo per inizializzare issfcm, in questo momento vanno indicati tutti i parametri richiesti dal metodo
    # issfcm ad esclusione dei dati da utilizzare che verranno passati seguendo il workflow di multiflow
    def __init__(self,
                 alpha,
                 fuzziness_coefficient=2,
                 max_iter=100,
                 stop_condition=('obj_delta', 0.001),
                 distance='euclidean',
                 output=None
                 ):
        super().__init__()
        self.alpha = alpha
        self.fuzziness_coefficient = fuzziness_coefficient
        self.max_iter = max_iter
        self.stop_condition = stop_condition
        self.distance = distance
        self.M = None
        self.prototypes = []
        self.output = output
        return

    '''
        Partially (incremental) fit the model
        X = ndarray (n_samples,n_features)
        y = array-like (classification target for all samples in X)
        classes = ndarray with all possible/known classes (default None)
        sample_weight not used
    '''

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        data = np.concatenate((X, np.reshape(y, (y.shape[0], 1))), axis=1)
        number_of_clusters = len(self.M[0]) if self.M is not None else 2
        b, F = get_b_F(X=data, n_clusters=number_of_clusters)
        membership_matrix, prototypes = issfcm(X=data, number_of_clusters=number_of_clusters,
                                               fuzziness_coefficient=self.fuzziness_coefficient, b=b,
                                               F=F, max_iter=self.max_iter, stop_condition=self.stop_condition,
                                               distance=self.distance,
                                               alpha=self.alpha, M=self.M, output=self.output)

        self.M = buildMemory(prototypes)
        self.prototypes = self.M[0]
        return self

    # predict classes for the passed data
    def predict(self, X=None):
        distanceMatrix = self.predict_proba(X)
        classes = list(self.prototypes[:, -1])
        samples = np.shape(X)[0]
        prediction = []
        a = np.argmin(distanceMatrix, axis=1)
        for i in range(samples):
            prediction.append(classes[a[i]])
        return prediction

    # produce la distance matrix tra i dati passati e i prototipi prodotti al passso precedente
    def predict_proba(self, X=None):
        return distance_matrix(X, self.prototypes[:, :-1])