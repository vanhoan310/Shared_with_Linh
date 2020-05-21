import numpy as np 
from scipy.spatial import distance
from sklearn.svm import LinearSVC
from bisect_louvain import * #import louvain clustering 

class SemiClassifier:
    def __init__(self):
        init = 0
         
    def predict(self, X_train, y_train, X_test, n_clusters):
        # compute the joint clustering, i.e, combine train and test data 
        y_full_semilouvain = semi_louvain_exact_K(X_train, y_train, X_test, n_clusters)
        y_full_semilouvain = np.array(y_full_semilouvain)
        n_samples = len(y_full_semilouvain)
        train_predict_labels = y_full_semilouvain[:len(y_train)]
        test_predict_labels  = y_full_semilouvain[len(y_train): n_samples]
        n_test = len(test_predict_labels)
        
        # compute distance matrix in train set to estimate the radius of being number
        train_dist = distance.cdist(X_train, X_train, 'euclidean')
        np.fill_diagonal(train_dist, train_dist[0,1]) #remove 0 from diag 

        # compute center and radius for each class in train set 
        center = {}
        radius = {}
        for c in np.unique(y_train):
            member = (y_train == c)
            center[c] = np.mean(X_train[member, ], axis = 0) # get center of class c 
            radius[c] = (np.min(train_dist[member, member]) + np.max(train_dist[member, member]))/4.0

        y_predict = np.array([0]*n_test)
        K = np.max(y_train) + 10
        for cl in np.unique(test_predict_labels):
            closest_class = -1
            train_with_label_cl = y_train[train_predict_labels==cl]
            if train_with_label_cl.size > 0: 
                center_new = np.mean(X_test[test_predict_labels==cl, ], axis=0)
                for c in np.unique(train_with_label_cl):
                    if np.linalg.norm(center_new - center[c]) < radius[c]:
                        closest_class = c

            if closest_class==-1:
                y_predict[test_predict_labels == cl] = K + 1 # new class 
                K = K + 1

        # train SVM on samples with old labels 
        clfSVM = LinearSVC(max_iter=1000000).fit(X_train, y_train)
        y_svm = clfSVM.predict(X_test) 
        y_predict[y_predict==0] = y_svm[y_predict==0]

        return y_predict



