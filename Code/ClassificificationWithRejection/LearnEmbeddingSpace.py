import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

class Embedding:
    
    def __init__(self, embedded_option = None, control_neighbor = None):
        if embedded_option is None:
            self.embedded_option = "PCA"
        else:
            self.embedded_option = embedded_option
        if control_neighbor is None:
            self.control_neighbor = 1
        else:
            self.control_neighbor = control_neighbor
             
    def _machingX2X(self, X, y, train_indices, n_components):
        X_temp = PCA(n_components=n_components).fit_transform(X)
        clf = NearestNeighbors(n_neighbors = self.control_neighbor+1).fit(X_temp[train_indices])
        distance_neigbours = clf.kneighbors(X_temp[train_indices])
        neigbours = distance_neigbours[1]
        neigbours = [list(x) for x in neigbours]
        neigbours_labels = [[y[train_indices[i]] for i in ind] for ind in neigbours]
        corrected_pairs = [1 if np.min(neigbours_labels[i]) == np.max(neigbours_labels[i]) else 0 for i in range(len(train_indices))]
        proprotion_corrected_pairs = np.sum(corrected_pairs)/len(y[train_indices])
        return proprotion_corrected_pairs
    
    def converter(self, X, y, train_indices):
        if self.embedded_option is not "PCA":
            raise NameError('"embedded_option" must be "PCA"')  
        set_components = range(1, np.min((X).shape))
        maching_probas = [self._machingX2X(X, y, train_indices, n_components) for n_components in set_components]
        max_n_components = np.max(maching_probas)
        opt_n_components = set_components[maching_probas.index(max_n_components)]
        return PCA(n_components=opt_n_components).fit_transform(X)
          