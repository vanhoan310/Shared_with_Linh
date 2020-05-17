import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
import Comparison
import sys

import matplotlib
import matplotlib.pyplot as plt
from SplitData import Spliter
from bisect_louvain import * #import louvain clustering 
from SemiClassifier import SemiClassifier

for prefixFileName in ["pollen", "patel", "baron"]:
# for prefixFileName in ["pollen"]:
    print("===========================================================================")
    print("===========================================================================")
    fileName = "Data/" + prefixFileName + "-prepare-log_count_100pca.csv"
    df = pd.read_csv(fileName)
    Xy= df.values
    X= Xy[:,1:]
    y= Xy[:,0].astype(int)
    for left_out_proportion in [0.0, 0.2, 0.5, 0.9]:
    # for left_out_proportion in [0.5]:
        # print("===================xxxxxxxxxxxxxxxxxxxxxxx================================")
        print("Data: ", prefixFileName, ", left_out_proportion = ", left_out_proportion)
        for data_seed in range(5):
            proportion_unknown = 0.2
            
            spl =  Spliter(proportion_unknown = proportion_unknown, left_out_proportion = left_out_proportion, random_seed = data_seed)
            train_indices, test_indices, unknown_classes = spl.Split(X, y)
            
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]

            k1 = len(set(y_test))
            k2 = len(set(y))
            
            y_louvain = louvain_exact_K(X_test, k1)

            # y_full_louvain = louvain_exact_K(X[train_indices+test_indices], k2)
            # y_full_louvain = y_full_louvain[len(y_train): len(y)]

            # just joint clustering 
            y_full_semilouvain = semi_louvain_exact_K(X_train, y_train, X_test, k2)
            y_full_semilouvain = y_full_semilouvain[len(y_train): len(y)]

            # joint clustering + SVM 
            clf = SemiClassifier()
            y_predict = clf.predict(X_train, y_train, X_test, k2 + 1)

            print("Louvain on test set: ", adjusted_rand_score(y_louvain, y_test))
            # print("Louvain on full set: ", adjusted_rand_score(y_full_louvain, y_test))
            print("Semi-Louvain ARI   : ", adjusted_rand_score(y_full_semilouvain, y_test))
            print("Semi-LouSVM  ARI   : ", adjusted_rand_score(y_predict, y_test))

            # print("Train class: ", np.unique(y_train))
            # print("New class: ", set(y_test).difference(set(y_train)))
            # print(y_test)
            # print(y_predict)
            print("========================================================")
            
            
            
            
