# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:40:26 2020

@author: v3_gammaTest
"""

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
import math
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import operator
import random
from random import choices
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import pandas as pd

from eval_bisect_louvain import louvain_exact_K #import louvain clustering 

def LearnEpsilon(k, X):
    Knn_temp = NearestNeighbors(n_neighbors=k+1)
    Knn_temp.fit(X)
    distances = Knn_temp.kneighbors(X)[0]
    extrated_distances = [np.sort(x)[-1] for x in distances]
    return [np.min(extrated_distances), np.max(extrated_distances), np.mean(extrated_distances)]
#%% Algorithm 1 
def SequentialRadiusNeighborsClassifier(epsilon, X_train, X_test, Y_train, add):
    X_train_temp =  np.copy(X_train)
    Y_train_temp =  np.copy(Y_train)
    Reps = RadiusNeighborsClassifier(radius=epsilon, weights='distance')
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
            Y_predict[optimal_test] = y_predict 
            if add == 1:
                X_train_temp = np.append(X_train_temp, [X_test[optimal_test]], axis =0)
                Y_train_temp = np.append(Y_train_temp, [y_predict], axis =0)
        else:
            y_predict = max(Y_current) + 1
            Y_current.append(y_predict)
            Y_predict[optimal_test] = y_predict 
            X_train_temp = np.append(X_train_temp, [X_test[optimal_test]], axis =0)
            Y_train_temp = np.append(Y_train_temp, [y_predict], axis =0)
    return Y_predict
#%% 
def SplitData(X, Y, random_seed=-1):
    if random_seed != -1:
        print("generate train and test data with seed number: ", random_seed)
        random.seed(random_seed) # use user random seed
    Y_all_labels = list(set(Y))
    leave_out_classes = list(set(choices(Y_all_labels, k=int(len(Y_all_labels)/2))))
    print(leave_out_classes)
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

def matchLabel(Y_labels, Y_ref):
    # this function changes clusters' label of Y_labels s.t ARI(Y_labels, Y_ref) is maximal.
    Y_labels_set = np.unique(Y_labels)
    Y_ref_set = np.unique(Y_ref)
    Y_result = np.copy(Y_labels)
    for x in Y_labels_set:
        arg_index = -1
        max_value = -1
        for y in Y_ref_set:
            setA = (Y_labels == x)
            setB = (Y_ref == y)
            sumAB = np.sum(setA*setB)
            if sumAB > max_value:
                max_value = sumAB 
                arg_index = y
        Y_result[Y_labels==x] = arg_index #replace x in Y_labels by arg_index
    return Y_result

#%%  # dataset: "pollen", "baron", "muraro", "patel", "xin", "zeisel"
for prefixFileName in ["pollen", "baron", "muraro", "patel", "xin", "zeisel"]:  
    fileName = "../Data/" + prefixFileName + "-prepare-log_count_100pca.csv"
    choice = "mean" #choice = "mean", choice = "min", choice = "max"
    add = 1
    data_seed = int(sys.argv[1])
    real_random_number = int(1000000*random.random()) # get real random number for cross validation
    times = 1   
    df = pd.read_csv(fileName)
    XY= df.values
    X= XY[:,1:]
    Y= XY[:,0].astype(int)
    epsilon_min, epsilon_max,  epsilon_mean = LearnEpsilon(1, X) 
    if choice == "min":
        epsilon_choice = epsilon_min
    if choice == "max":
        epsilon_choice = epsilon_max
    if choice == "mean":
        epsilon_choice = epsilon_mean
 #   for data_seed in [see for see in range(10)]: 
    ARI_merge_clusters = []
    ARI_SequentialRadiusNeighborsClassifier = []
    ARI_louvain = []
    for repeat_time in range(times):
        data_seed += repeat_time
        print("data_seed:", data_seed)
        # print("data_seed: ", data_seed)
        X_train, X_test, Y_train, Y_test = SplitData(X, Y, random_seed=data_seed)
        # run Louvain algorithm
        n_clusters = int(len(np.unique(Y_test)))
        print("n_clusters: ", n_clusters)
        louvain_labels = louvain_exact_K(X_test, n_clusters)
        ARI_louvain.append(adjusted_rand_score(louvain_labels, Y_test))

        # Run internal cross validation to choose K and epsilon   
        Y_predict = SequentialRadiusNeighborsClassifier(epsilon_choice, X_train, X_test, Y_train, add)
        ARI_Srn_repeat_time = adjusted_rand_score(Y_predict, Y_test)
        ARI_SequentialRadiusNeighborsClassifier.append(ARI_Srn_repeat_time)
        ## Merge clusters
        Y_predict = SequentialRadiusNeighborsClassifier(epsilon_choice, X_train, X_test, Y_train, add)
        print("merge label to match the ground truth")
        Y_predict = matchLabel(Y_predict, Y_test)
        ARI_Srn_merge_clusters_repeat_time = adjusted_rand_score(Y_predict, Y_test)
        ARI_merge_clusters.append(ARI_Srn_merge_clusters_repeat_time)
        print("-------------------------------------------------------------------")
        print("ARI_Srn_repeat_time:", ARI_Srn_repeat_time)
        print("ARI_Srn_merge_clusters_repeat:", ARI_Srn_merge_clusters_repeat_time)
        print("===================================================================")
    print("ARI_Srn               :", (ARI_SequentialRadiusNeighborsClassifier))
    print("ARI_Srn_merge_clusters:", ARI_merge_clusters)
#    print("ARI_louvain           :", ARI_louvain)
    df = pd.DataFrame(data= {'ARI_Srn': ARI_SequentialRadiusNeighborsClassifier, 'ARI_Srn_merge_clusters': ARI_merge_clusters})
    df.to_csv("output/CV_0_Gamma_0/" +prefixFileName+ "ARI_dataseed_"+str(data_seed)+"_add_"+str(add)+"_eps_"+choice+".csv", index=False)
    df_lv = pd.DataFrame(data={'ARI_louvain': ARI_louvain})
    df_lv.to_csv("output/CV_0_Gamma_0/" +prefixFileName+ "ARI_louvain_dataseed_"+str(data_seed)+".csv", index=False)

    # results_save = [[fileName]]
    # results_save += [["ARI_Knn_repeat_time"] + ARI_KNeighborsClassifier]
    # results_save += [["ARI_Srn_repeat_time"] + ARI_SequentialRadiusNeighborsClassifier]
    # results_save += [["ARI_Knn_mean"] + [np.mean(ARI_KNeighborsClassifier)]]
    # results_save += [["ARI_Srn_mean"] + [np.mean(ARI_SequentialRadiusNeighborsClassifier)]]
    # res_file = "Knn_vs_Srn_%i_%s" %(times, fileName)
    # file = open(res_file, "w")
    # file.writelines("%s\n" %listToStringWithoutBrackets(line) for line in results_save)
    # file.close()        
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
