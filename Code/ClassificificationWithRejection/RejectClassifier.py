import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC

class Classifier:
    
    def __init__(self, predictive_alg = None,  threshold_rejection = None):
        if predictive_alg is None:
            self.predictive_alg = "SVM"
        else:
            self.predictive_alg = predictive_alg
        if threshold_rejection is None:
            self.threshold_rejection = 0.7
        else:
             self.threshold_rejection =  threshold_rejection
         
    def _fit(self, X, y):
        if self.predictive_alg == "SVM":
            return svm.SVC(probability=True, max_iter=10000, gamma = 'scale').fit(X, y) 
        elif self.predictive_alg == "linearSVM":
            return LinearSVC(max_iter=1000000).fit(X, y)
        else:
            raise NameError('"predictive_alg" must be "SVM" or "linearSVM"')            

    def predict(self, X_train, y_train, X_test):
        self.new_classes = int(10*np.max(y_train) + 1) 
        clf = self._fit(X_train, y_train)
        probs_max = [np.max(x) for x in clf.predict_proba(X_test)]
        y_predict = list(clf.predict(X_test))
        y_predict = [y_predict[i] if probs_max[i] >= self.threshold_rejection else self.new_classes for i in range(X_test.shape[0])]
        return y_predict
