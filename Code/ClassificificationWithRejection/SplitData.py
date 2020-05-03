import numpy as np
from operator import itemgetter
from random import choices

from sklearn.model_selection import train_test_split

class Spliter:
    
    def __init__(self, proportion_unknown = None,  left_out_proportion = None, random_seed = None):
        if proportion_unknown is None:
            self.proportion_unknown = 0
        else:
            self.proportion_unknown = proportion_unknown
        if left_out_proportion is None:
            self.left_out_proportion = 0
        else:
             self.left_out_proportion =  left_out_proportion 
        if random_seed is None:
            self.random_seed = 0
        else:
             self.random_seed =  random_seed 

    def Split(self, X, y):
        labels_set = list(set(y))
        labels_counts = [len(list(np.where(y == y_label)[0])) for y_label in labels_set]
        indices, L_sorted = zip(*sorted(enumerate(labels_counts), key=itemgetter(1)))
        y_minor_labels = indices[:int(self.proportion_unknown*len(labels_set))]
        unknown_classes = list(set(choices(y_minor_labels, k=int(len(y_minor_labels)))))
        ids = [i for i in range(X.shape[0]) if y[i] not in unknown_classes]
        if self.left_out_proportion > 0:
            train_indices, test_indices = train_test_split(ids, test_size = self.left_out_proportion, random_state = self.random_seed)
        else:
            train_indices = ids
        test_indices = [i for i in range(X.shape[0]) if i not in train_indices]
        known_classes = list(set(y[train_indices]))
        actual_unknown_classes = [i for i in labels_set if i not in known_classes]
        return [train_indices, test_indices, actual_unknown_classes]