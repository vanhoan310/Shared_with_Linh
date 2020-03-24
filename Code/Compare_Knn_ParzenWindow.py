# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:40:26 2020

@author: nguye
"""

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
import math
import numpy as np
from sklearn.model_selection import train_test_split
import operator
from random import choices
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import pandas as pd

##%%
##Generate data
#X, Y = make_classification(
#                class_sep= 3,
#                flip_y=0.05,
#                n_samples=2000,
#                n_features=6,
#                n_informative=3,
#                n_redundant=0,
#                n_repeated=0,
#                n_classes=6,
#                n_clusters_per_class=1,
#                weights=None,
#                hypercube=True,
#                shift=0.0,
#                scale=1.0,
#                shuffle=True,
#                random_state=1)
#%%   
def LearnEpsilon(k, X):
    Knn_temp = NearestNeighbors(n_neighbors=k+1)
    Knn_temp.fit(X)
    distances = Knn_temp.kneighbors(X)[0]
    extrated_distances = [np.sort(x)[-1] for x in distances]
    return np.mean(extrated_distances)
#%%
def SequentialRadiusNeighborsClassifier(epsilon, X_train, X_test, Y_train):
    X_train_temp =  np.copy(X_train)
    Y_train_temp =  np.copy(Y_train)
    Reps = RadiusNeighborsClassifier(radius=epsilon)
    test_size = len(X_test)
    Y_predict = [-1 for x in range(test_size)]
    Y_current = list(set(Y_train))
    test_index = [x for x in range(test_size)]
    for test_time in range(test_size):
        Knn_temp = NearestNeighbors(n_neighbors=1)
        Knn_temp.fit(X_train_temp)
        min_distances = Knn_temp.kneighbors(X_test[test_index])[0]
        min_distances = [np.mean(x) for x in min_distances]
        optimal_indice = min_distances.index(min(min_distances))
        optimal_test = test_index[optimal_indice]
        test_index.remove(optimal_test)
        Reps.fit(X_train_temp, Y_train_temp)
        predict_set = Reps.radius_neighbors(X_test[optimal_test].reshape(1, -1))[1]
        predict_set = predict_set[0]
        if predict_set.size > 0:
            y_predict = Reps.predict(X_test[optimal_test].reshape(1, -1))
            y_predict = y_predict[0]
        else:
            y_predict = max(Y_current) + 1
            Y_current.append(y_predict)
        Y_predict[optimal_test] = y_predict 
        X_train_temp = np.append(X_train_temp, [X_test[optimal_test]], axis =0)
        Y_train_temp = np.append(Y_train_temp, [y_predict], axis =0)
    return Y_predict
#%% 
def SplitData(X, Y):
    Y_all_labels = list(set(Y))
    leave_out_classes = list(set(choices(Y_all_labels, k=int(len(Y_all_labels)/2))))
    X_train_all, X_test_all, Y_train_all, Y_test_all = train_test_split(X, Y, test_size = 0.05, random_state = 0)
    indices_leave_out = []
    for leave_out_class in leave_out_classes:
        indices_leave_out += list(np.where(Y_train_all == leave_out_class)[0])
    X_train = np.delete(X_train_all, indices_leave_out, axis=0)
    Y_train = np.delete(Y_train_all, indices_leave_out, axis=0)
    X_test = np.append(X_test_all, X_train_all[indices_leave_out], axis=0)
    Y_test = np.append(Y_test_all, Y_train_all[indices_leave_out], axis=0)
    return [X_train, X_test, Y_train, Y_test]
#%%    
#%%
def listToStringWithoutBrackets(list1):
    return str(list1).replace('[','').replace(']','')    
#%% 
for fileName in ["pollen-prepare-log_count_100pca.csv"]:   
    times = 20    
    k_set =  [k+1 for k in range(10, 50)] 
    df = pd.read_csv(fileName)
    XY= df.values
    X= XY[:,1:]
    Y= XY[:,0]
    epsilon_set = [LearnEpsilon(k, X) for k in k_set]
    ARI_KNeighborsClassifier = []
    ARI_SequentialRadiusNeighborsClassifier = []
#    Size_SequentialRadiusNeighborsClassifier = []
    for repeat_time in range(times):
        X_train, X_test, Y_train, Y_test = SplitData(X, Y)
# Run internal cross validation to choose K and epsilon   
        ARI_Knn_repeat = [0 for k in k_set]
        ARI_Srn_repeat = [0 for k in k_set]
        for repeat_val in range(times):
#            Size_Srn_repeat = [] 
            print("We are in repeat_time and repeat_val:", repeat_time +1, repeat_val +1)
            X_train_val, X_test_val, Y_train_val, Y_test_val = SplitData(X_train, Y_train)
            for ind in range(len(k_set)):
                Knn = KNeighborsClassifier(n_neighbors=k_set[ind], algorithm='auto')
                Knn.fit(X_train_val, Y_train_val)
                Y_predict_Knn = Knn.predict(X_test_val)
                Y_predict = SequentialRadiusNeighborsClassifier(epsilon_set[ind], X_train_val, X_test_val, Y_train_val)
                ARI_Knn_repeat[ind] += adjusted_rand_score(Y_predict_Knn, Y_test_val)               
                ARI_Srn_repeat[ind] += adjusted_rand_score(Y_predict, Y_test_val)
        k_optimal = k_set[ARI_Knn_repeat.index(max(ARI_Knn_repeat))] 
        epsilon_optimal = epsilon_set[ARI_Srn_repeat.index(max(ARI_Srn_repeat))]  
        Knn = KNeighborsClassifier(n_neighbors=k_optimal, algorithm='auto')
        Knn.fit(X_train, Y_train)
        Y_predict_Knn = Knn.predict(X_test)
        Y_predict = SequentialRadiusNeighborsClassifier(epsilon_optimal, X_train, X_test, Y_train)
        ARI_Knn_repeat_time = adjusted_rand_score(Y_predict_Knn, Y_test)
        ARI_KNeighborsClassifier.append(ARI_Knn_repeat_time)
        ARI_Srn_repeat_time = adjusted_rand_score(Y_predict, Y_test)
        ARI_SequentialRadiusNeighborsClassifier.append(ARI_Srn_repeat_time)
        print("ARI_Knn_repeat_time:", ARI_Knn_repeat_time)
        print("ARI_Srn_repeat_time:", ARI_Srn_repeat_time)
    print("ARI_Srn_mean:", np.mean(ARI_KNeighborsClassifier))
    print("ARI_Srn_mean:", np.mean(ARI_SequentialRadiusNeighborsClassifier))
    results_save = [[fileName]]
    results_save += [["ARI_Knn_repeat_time"] + ARI_KNeighborsClassifier]
    results_save += [["ARI_Srn_repeat_time"] + ARI_SequentialRadiusNeighborsClassifier]
    results_save += [["ARI_Knn_mean"] + [np.mean(ARI_KNeighborsClassifier)]]
    results_save += [["ARI_Srn_mean"] + [np.mean(ARI_SequentialRadiusNeighborsClassifier)]]
    res_file = "Knn_vs_Srn_%i_%s" %(times, fileName)
    file = open(res_file, "w")
    file.writelines("%s\n" %listToStringWithoutBrackets(line) for line in results_save)
    file.close()        
#            print("ARI score of KnnClassifier is", ARI_Knn)
#            print("ARI score of SrnClassifier is", ARI_Srn)
#            Size_label = len(list(set(Y_predict)))
#            print("# predicted labels given by SrnClassifier is", Size_label) 
#            Size_Srn_repeat.append(Size_label)
#            print("# new labels in the test set is", len(Y_all_labels) - len(list(set(Y_train))))
#            print("# actual labels in the data set is", len(Y_all_labels))
#            print("------------------------------------------------------")
#        plt.plot(K_set, ARI_KNeighborsClassifier, color='b')
#        plt.plot(K_set, ARI_SequentialRadiusNeighborsClassifier, color='r')
#        plt.xlabel('K')
#        plt.ylabel('ARI')
#        plt.title('ARI of Knn (blue) and Srn (red)')
#        plt.savefig("ARI_Knn_Srn.pdf", bbox_inches='tight')
#        plt.show()
#        plt.plot(K_set, Size_SequentialRadiusNeighborsClassifier, color='r')
#        plt.xlabel('K')
#        plt.ylabel('# of labels')
#        plt.title('# of labels produced by Srn.pdf')
#        plt.savefig("Size_Srn.pdf", bbox_inches='tight')
#        plt.show()
