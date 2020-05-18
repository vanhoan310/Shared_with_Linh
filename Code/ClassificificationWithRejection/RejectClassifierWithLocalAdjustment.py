import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import RadiusNeighborsClassifier


class ClassifierWithLocalAdjustment:
    
    def __init__(self, predictive_alg = None, control_neighbor = None, 
                 filter_proportion = None, threshold_rejection = None,
                 sequential_order = None):
        if predictive_alg is None:
            self.predictive_alg = "SVM"
        else:
            self.predictive_alg = predictive_alg
        if control_neighbor is None:
            self.control_neighbor = 1
        else:
            self.control_neighbor = control_neighbor 
        if filter_proportion is None:
            self.filter_proportion = 0
        else:
            self.filter_proportion = filter_proportion
        if threshold_rejection is None:
            self.threshold_rejection = 0.7
        else:
             self.threshold_rejection =  threshold_rejection
        if sequential_order is None:
            self.sequential_order = "density"
        else:
            self.sequential_order  = sequential_order

    def _epsilonX2X(self, X, y):
        Knn_clf = NearestNeighbors(n_neighbors = self.control_neighbor + 1).fit(X)
        distancesNeighbours = Knn_clf.kneighbors(X)
        neighbours = [list(x) for x in distancesNeighbours[1]]
        neigbours_labels = [[y[i] for i in ind] for ind in neighbours]
        extrated_distances = [np.max(x) for x in distancesNeighbours[0]]
        corrected_pairs = [1 if np.min(neigbours_labels[i]) == np.max(neigbours_labels[i]) else 0 for i in range(y.shape[0])]
        extrated_distances_false = [extrated_distances[i] for i in range(y.shape[0]) if corrected_pairs[i] ==0]
        if len(extrated_distances_false) > 0:
             return np.percentile(extrated_distances_false, self.filter_proportion)
        else:
            extrated_distances_true = [extrated_distances[i] for i in range(y.shape[0]) if corrected_pairs[i] ==1]
            return np.max(extrated_distances_true) 
        
    def _learn_sequential_order(self, X_train, y_train, X_test):
        if self.sequential_order == "density":
            Knn_temp = NearestNeighbors(n_neighbors= 2).fit(np.append(X_train, X_test, axis =0))
            max_distances_test = Knn_temp.kneighbors(X_test)[0]
            max_distances_test = [np.max(x) for x in max_distances_test]
            return np.argsort(max_distances_test)
        elif self.sequential_order == "increasing_confidence": 
            clf = self.fit(X_train, y_train)
            predict_proba = clf.predict_proba(X_test)
            return np.argsort(predict_proba)
        elif self.sequential_order == "deccreasing_confidence": 
            clf = self.fit(X_train, y_train)
            predict_proba = clf.predict_proba(X_test)
            return np.argsort(-predict_proba)
         
    def _fit(self, X, y):
        if self.predictive_alg == "SVM":
            return svm.SVC(probability=True, max_iter=10000, gamma = 'scale').fit(X, y) 
        elif self.predictive_alg == "linearSVM":
            return LinearSVC(max_iter=1000000).fit(X, y)
        else:
            raise NameError('"predictive_alg" must be "SVM" or "linearSVM"')
            
    def _transductive_classifier(self, X_train, y_train, test_instance):
        clf = RadiusNeighborsClassifier(radius=self.epsilon, weights='distance').fit(X_train, y_train)        
        predict_set = clf.radius_neighbors(test_instance.reshape(1, -1))[1]
        predict_set = list(predict_set[0])
        if len(predict_set) > 0:
            X_train_local, y_train_local = X_train[predict_set], y_train[predict_set]
            if np.min(y_train_local) == np.max(y_train_local):
                prediction = y_train_local[0] 
            else:
                clf = self._fit(X_train_local, y_train_local) 
                if np.max(clf.predict_proba(test_instance.reshape(1, -1))) < self.threshold_rejection:
                    prediction = self.new_classes 
                else:
                    prediction = clf.predict(test_instance.reshape(1, -1))[0]
        else:
            prediction = self.new_classes
        return prediction

    def predict(self, X_train, y_train, X_test):
        X_train_temp =  np.copy(X_train)
        y_train_temp =  np.copy(y_train)
        self.new_classes = int(10*np.max(y_train) + 1) 
        self.epsilon = self._epsilonX2X(X_train, y_train)
        order = self._learn_sequential_order(X_train, y_train, X_test)
        clf = self._fit(X_train, y_train)
        test_size = X_test.shape[0]
        y_predict = [-1 for i in range(test_size)]
        for test in range(test_size):
            if np.max(clf.predict_proba(X_test[order[test]].reshape(1, -1))) < self.threshold_rejection:
                      prediction = self._transductive_classifier(X_train_temp, y_train_temp, X_test[order[test]]) 
            else:
                     prediction = clf.predict(X_test[order[test]].reshape(1, -1))[0]            
            X_train_temp = np.append(X_train_temp, [X_test[order[test]]], axis =0)
            y_train_temp = np.append(y_train_temp, [prediction], axis =0)
            y_predict[order[test]] = prediction 
        return y_predict
        
#if __name__ == "__main__":
#    pass    
