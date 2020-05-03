import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
import Comparison
import sys

import matplotlib
import matplotlib.pyplot as plt
from bisect_louvain import * #import louvain clustering 
from SplitData import Spliter

for prefixFileName in ["pollen", "patel", "baron"]:
    print("===========================================================================")
    print("===========================================================================")
    fileName = "Data/" + prefixFileName + "-prepare-log_count_100pca.csv"
    df = pd.read_csv(fileName)
    Xy= df.values
    X= Xy[:,1:]
    y= Xy[:,0].astype(int)
    for left_out_proportion in [0.0, 0.2, 0.5, 0.9]:
        print("===================================================")
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
            
            y_full_louvain = louvain_exact_K(X[train_indices+test_indices], k2)
            y_full_louvain = y_full_louvain[len(y_train): len(y)]
            
            y_full_semilouvain = semi_louvain_exact_K(X_train, y_train, X_test, k2)
            y_full_semilouvain = y_full_semilouvain[len(y_train): len(y)]
            
            print("Louvain on test set: ", adjusted_rand_score(y_louvain, y_test))
            print("Louvain on full set: ", adjusted_rand_score(y_full_louvain, y_test))
            print("Semi-Louvain ARI   : ", adjusted_rand_score(y_full_semilouvain, y_test))
            print("--------------------------------------------------")
            
            
            
            
