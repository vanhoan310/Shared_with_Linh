"""
Created on Sat Mar  7 18:40:26 2020

@author: v3_gammaTest
"""

#from sklearn.datasets import make_classification
#import matplotlib.pyplot as plt
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
#import math
import sys
import numpy as np
from sklearn.model_selection import train_test_split
#import operator
import random
from random import choices
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
#from scipy.spatial import distance
import pandas as pd
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
#from scipy.stats import bernoulli
#from sklearn.preprocessing import MinMaxScaler

from eval_bisect_louvain import louvain_exact_K #import louvain clustering 
#%%
def LearnEpsilon(X, choice):
    Knn_temp = NearestNeighbors(n_neighbors = 4)
    Knn_temp.fit(X)
    distances = Knn_temp.kneighbors(X)[0]
    extrated_distances = np.array([0.5*np.sort(x)[-2]+ 0.5*np.sort(x)[-1] for x in distances])
#    extrated_distances = np.sort(extrated_distances)[int(len(extrated_distances)*0.05): -int(len(extrated_distances)*0.05)]
    if  type(choice) == str:
        if choice == "mean":
            return np.mean(extrated_distances)
    else:
        return choice*np.min(extrated_distances) + (1-choice)*np.max(extrated_distances)
#%% Algorithm 1 
def SequentialRadiusNeighborsClassifier(epsilon, X_train, X_test, Y_train, add, alg):
#    size_train = len(Y_train)
    X_train_temp =  np.copy(X_train)
    Y_train_temp =  np.copy(Y_train)
    test_size = len(X_test)
    Y_predict = [-1 for x in range(test_size)]
    Y_current = list(set(Y_train))
    test_index = [x for x in range(test_size)]
    new_indices = []
    epsilon_update = epsilon
#    epsilon_update = updateEpsilon(distances, test_index, choice)
    for test_time in range(test_size):
        Knn_temp = NearestNeighbors(n_neighbors=1)
        Knn_temp.fit(X_train_temp)
        min_distances = Knn_temp.kneighbors(X_test[test_index])[0]
        min_distances = [np.mean(x) for x in min_distances]
        optimal_indice = min_distances.index(min(min_distances))
        optimal_test = test_index[optimal_indice]
        clf = RadiusNeighborsClassifier(radius=epsilon_update, weights='distance').fit(X_train_temp, Y_train_temp)        
        predict_set = clf.radius_neighbors(X_test[optimal_test].reshape(1, -1))[1]
        predict_set = list(predict_set[0])
        if len(predict_set)> 0:
            if min(Y[predict_set]) == max(Y[predict_set]):
                     y_predict =  min(Y[predict_set]) 
            else:
                if alg == "srnc":
                    y_predict = clf.predict(X_test[optimal_test].reshape(1, -1))
                    y_predict = y_predict[0]
                else:                        
                    if alg == "svm":
                        clf = svm.SVC().fit(X[predict_set], Y[predict_set])
                    if alg == "LinearSVC":
                        # clf = LinearSVC(max_iter=10000).fit(X[predict_set], Y[predict_set])
                        clf = LinearSVC().fit(X[predict_set], Y[predict_set])
                    if alg == "dt":
                        clf = DecisionTreeClassifier().fit(X[predict_set], Y[predict_set])
                    if alg == "rf":
                        clf = RandomForestClassifier(n_estimators=10).fit(X[predict_set], Y[predict_set])
                    if alg == "gb":
                        clf = GradientBoostingClassifier(n_estimators=10).fit(X[predict_set], Y[predict_set])
                    if alg == "lr":
                        clf = LogisticRegression(max_iter=10000).fit(X[predict_set], Y[predict_set])
                    if alg == "mlp":
                        clf = MLPClassifier().fit(X[predict_set], Y[predict_set])
                    y_predict = clf.predict(X_test[optimal_test].reshape(1, -1))
                    y_predict = y_predict[0]
            if add ==1:
                X_train_temp = np.append(X_train_temp, [X_test[optimal_test]], axis =0)
                Y_train_temp = np.append(Y_train_temp, [y_predict], axis =0)
        else:
            y_predict = max(Y_current) + 1
            Y_current.append(y_predict)
            X_train_temp = np.append(X_train_temp, [X_test[optimal_test]], axis =0)
            Y_train_temp = np.append(Y_train_temp, [y_predict], axis =0)
            new_indices.append(optimal_test)
#            epsilon_update = updateEpsilon(distances, test_index, choice)
        Y_predict[optimal_test] = y_predict 
        test_index.remove(optimal_test)
    return Y_predict
#%% 
def SplitData(X, Y, random_seed=-1):
    if random_seed != -1:
        print("generate train and test data with seed number: ", random_seed)
        random.seed(random_seed) # use user random seed
    Y_all_labels = list(set(Y))
    leave_out_classes = list(set(choices(Y_all_labels, k=int(len(Y_all_labels)/2))))
    print(leave_out_classes)
    X_train_all, X_test_all, Y_train_all, Y_test_all = train_test_split(X, Y, test_size = 0.1, random_state = 0)
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
for prefixFileName in [1]:  
    datasets = ["pollen", "baron", "muraro", "patel", "xin", "zeisel"]
    prefixFileName = datasets[int(sys.argv[2])]
    print("Dataset: ", prefixFileName)
    fileName = "../Data/" + prefixFileName + "-prepare-log_count_100pca.csv"
    # choice = "mean",  choice = L \in [0,1]
    # alg = "srnc", alg = "svm", alg = "dt",  alg = "lr"
    choice = 0 #choice = 1 --> "min", choice = 0 --> "max",
    add = 1
#    alg = "svm"
    alg = "LinearSVC"
    data_seed = int(sys.argv[1])
    times = 1  
    df = pd.read_csv(fileName)
    XY= df.values
    X= XY[:,1:]
    Y= XY[:,0].astype(int)
    epsilon_choice = LearnEpsilon(X, choice) 
 #   for data_seed in [see for see in range(10)]: 
    ARI_merge_clusters = []
    ARI_SequentialRadiusNeighborsClassifier = []
    ARI_louvain = []
    for repeat_time in range(times):
        data_seed += repeat_time
#        data_seed = repeat_time
        print("data_seed:", data_seed)
        # print("data_seed: ", data_seed)
        X_train, X_test, Y_train, Y_test = SplitData(X, Y, random_seed=data_seed)
        # run Louvain algorithm
        n_clusters = int(len(np.unique(Y_test)))
        print("n_clusters: ", n_clusters)
        louvain_labels = louvain_exact_K(X_test, n_clusters)
        ARI_louvain.append(adjusted_rand_score(louvain_labels, Y_test))
        print("Louvain done!")
        #Run internal cross validation to choose K and epsilon   
        Y_predict_src = SequentialRadiusNeighborsClassifier(epsilon_choice, X_train, X_test, Y_train, add, alg)
        ARI_Srn_repeat_time = adjusted_rand_score(Y_predict_src, Y_test)
        ARI_SequentialRadiusNeighborsClassifier.append(ARI_Srn_repeat_time)
        ## Merge clusters
        Y_predict_merge_clusters = SequentialRadiusNeighborsClassifier(epsilon_choice, X_train, X_test, Y_train, add, alg)
        print("merge label to match the ground truth")
        Y_predict_merge_clusters = matchLabel(Y_predict_merge_clusters, Y_test)
        ARI_Srn_merge_clusters_repeat_time = adjusted_rand_score(Y_predict_merge_clusters, Y_test)
        ARI_merge_clusters.append(ARI_Srn_merge_clusters_repeat_time)
        print("-------------------------------------------------------------------")
        print("number_of_true_class:", len(set(Y)))
        print("number_of_test_class:", len(set(Y_test)))
        print("number_of_cluster_Srn_repeat_time:", len(set(Y_predict_src)))
        print("number_of_cluster_Srn_merge_clusters_repeat_time:", len(set(Y_predict_merge_clusters)))
        print("-------------------------------------------------------------------")
        #print("ARI_Srn_repeat_time:", ARI_Srn_repeat_time)
        #print("ARI_Srn_merge_clusters_repeat:", ARI_Srn_merge_clusters_repeat_time)
        print("===================================================================")
    print("ARI_Srn               :", (ARI_SequentialRadiusNeighborsClassifier))
    print("ARI_Srn_merge_clusters:", ARI_merge_clusters)
    print("ARI_louvain           :", ARI_louvain)
    print("fileName = ", str(prefixFileName), "choice = ", str(choice), "alg = ", str(alg), "add = ", str(add))
    print("=================================DONE==================================")
    df = pd.DataFrame(data= {'ARI_Srn': ARI_SequentialRadiusNeighborsClassifier, 'ARI_Srn_merge_clusters': ARI_merge_clusters})
    df.to_csv("output/CV_0_Gamma_0/" +prefixFileName+ "_ARI_dataseed_"+str(data_seed)+"_add_"+str(add)+"_eps_"+str(choice)+"_alg_"+str(alg)+".csv", index=False)
    df_lv = pd.DataFrame(data={'ARI_louvain': ARI_louvain})
    df_lv.to_csv("output/CV_0_Gamma_0/" +prefixFileName+ "_ARI_louvain_dataseed_"+str(data_seed)+".csv", index=False)
    

