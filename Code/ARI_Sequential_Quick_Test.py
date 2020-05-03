# -ARI_Sequential_Classification.py- coding: utf-8 -*-
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
from sklearn import linear_model
#from scipy.stats import bernoulli
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#from eval_bisect_louvain import louvain_exact_K #import louvain clustering 
#%%
def LearnEpsilon(k, X, choice):
#    alpha = 0.05
    Knn_temp = NearestNeighbors(n_neighbors = k + 1).fit(X)
    distances = Knn_temp.kneighbors(X)[0]
    extrated_distances = np.array([x[-1] for x in distances])
#    length = extrated_distances.shape[0]
#    filter_extrated_distances = np.sort(extrated_distances)[int(alpha*length*0.5): - int(alpha*length*0.5)]
    if  type(choice) == str:
        if choice == "mean":
            return np.mean(extrated_distances)
    else:
        return choice*np.min(extrated_distances) + (1-choice)*np.max(extrated_distances)
#%%
def MaxEpsilonXtest2X(X, X_test):
#    alpha = 0.05
    Knn_temp = NearestNeighbors(n_neighbors = 2).fit(X)
    distances = Knn_temp.kneighbors(X_test)[0]
    extrated_distances = np.array([x[-1] for x in distances])
#    length = extrated_distances.shape[0]
    return np.max(extrated_distances)        
#%%
def Classifier(train_size, new_classes, optimal_test, epsilon_choice, X_train_temp, X_test, Y_train_temp, alg):
    clf = RadiusNeighborsClassifier(radius=epsilon_choice, weights='distance').fit(X_train_temp, Y_train_temp)        
    predict_set = clf.radius_neighbors(X_test[optimal_test].reshape(1, -1))[1]
    predict_set = list(predict_set[0])
    if len(predict_set) > 0:
        if min(Y_train_temp[predict_set]) == max(Y_train_temp[predict_set]):
            return  [min(Y_train_temp[predict_set]), predict_set] 
        else:
            if alg == "srnc":
                y_predict = clf.predict(X_test[optimal_test].reshape(1, -1))        
            else:                        
                if alg == "svm":
                    clf = svm.SVC().fit(X_train_temp[predict_set], Y_train_temp[predict_set])
                if alg == "LinearSVC":
                    clf = LinearSVC(max_iter=500000).fit(X_train_temp[predict_set], Y_train_temp[predict_set])
                if alg == "sgd":
                    clf = linear_model.SGDClassifier().fit(X_train_temp[predict_set], Y_train_temp[predict_set])
                if alg == "dt":
                    clf = DecisionTreeClassifier().fit(X_train_temp[predict_set], Y_train_temp[predict_set])
                if alg == "rf":
                    clf = RandomForestClassifier(n_estimators=10).fit(X_train_temp[predict_set], Y_train_temp[predict_set])
                if alg == "gb":
                    clf = GradientBoostingClassifier(n_estimators=10).fit(X_train_temp[predict_set], Y_train_temp[predict_set])
                if alg == "lr":
                    clf = LogisticRegression(max_iter=1000).fit(X_train_temp[predict_set], Y_train_temp[predict_set])
                if alg == "mlp":
                    clf = MLPClassifier().fit(X_train_temp[predict_set], Y_train_temp[predict_set])
                y_predict = clf.predict(X_test[optimal_test].reshape(1, -1))
            return [y_predict[0], predict_set]
    else:
        return [new_classes, predict_set]  
#%%
def EnsembleClassifier(new_classes, optimal_test, Y_train_all_labels, epsilon_linspace, X_train_temp, X_test, Y_train_temp, alg):    
    y_predict_linspace = []    
    for i in range(len(epsilon_linspace)):
        y_predict, predict_set = Classifier(new_classes, optimal_test, epsilon_linspace[i], X_train_temp, X_test, Y_train_temp, alg) 
        if y_predict != new_classes and np.min(Y_train_temp[predict_set]) == np.max(Y_train_temp[predict_set]):
            return y_predict
        y_predict_linspace.append(y_predict)
    y_predict, num_y_predict = Counter(y_predict_linspace).most_common(1)[0] 
    return y_predict
#%% Algorithm 1 
def SequentialRadiusNeighborsClassifier(X, Y_all_labels, X_train, X_test, Y_train, add, alg):
#    size_train = len(Y_train)
    X_train_temp =  np.copy(X_train)
    Y_train_temp =  np.copy(Y_train)
    test_size = len(X_test)
    train_size = len(X_train)
    Y_predict = [-1 for x in range(test_size)]
    test_index = [x for x in range(test_size)]
    Y_train_all_labels = list(set(Y_train))
    new_classes = int(10*len(Y_all_labels) + 1) 
    for test_time in range(test_size):
        epsilon_choice = MaxEpsilonXtest2X(X, X_test[test_index])
        Knn_temp = NearestNeighbors(n_neighbors=2).fit(X)
        min_distances = Knn_temp.kneighbors(X_test[test_index])[0]
        min_distances = [np.max(x) for x in min_distances]
        optimal_indice = min_distances.index(np.min(min_distances))
        optimal_test = test_index[optimal_indice]
#        y_predict_linspace = Classifier(new_classes, optimal_test, epsilon_choice, X_train_temp, X_test, Y_train_temp, alg) 
        y_predict, predict_set = Classifier(train_size, new_classes, optimal_test, epsilon_choice, X_train_temp, X_test, Y_train_temp, alg) 
#        y_predict = EnsembleClassifier(train_size, new_classes, optimal_test, Y_train_all_labels, epsilon_linspace, X_train_temp, X_test, Y_train_temp, alg)
        if y_predict == new_classes:
            X_train_temp = np.append(X_train_temp, [X_test[optimal_test]], axis =0)
            Y_train_temp = np.append(Y_train_temp, [y_predict], axis =0)
#            print(np.where(Y_train_temp == new_classes)[0])
            new_classes += 1
        else:
            if add == 1:
                X_train_temp = np.append(X_train_temp, [X_test[optimal_test]], axis =0)
                Y_train_temp = np.append(Y_train_temp, [y_predict], axis =0)
#                epsilon_linspace = np.append(epsilon_linspace, [np.min(min_distances)], axis =0)
#            epsilon_update = updateEpsilon(distances, test_index, choice)
        Y_predict[optimal_test] = y_predict 
        test_index.remove(optimal_test)
#    sys.exit()
    return Y_predict
#%% 
def SplitData(X, Y, Y_all_labels, random_seed=-1):
    if random_seed != -1:
        print("generate train and test data with seed number: ", random_seed)
        random.seed(random_seed) # use user random seed
    unknown_classes = list(set(choices(Y_all_labels, k=int(0.3*len(Y_all_labels)))))
    print(unknown_classes)
    ids = np.array([i for i in range(X.shape[0])])
    train_indices, test_indices = train_test_split(ids, test_size = 0.1, random_state = 0)
    leave_out_indices = [i for i in train_indices if Y[i] in unknown_classes]
    train_indices = [i for i in train_indices if i not in leave_out_indices]
    test_indices = [i for i in ids if i not in train_indices]
#    sys.exit()
    return [train_indices, test_indices, unknown_classes]
#%%
def listToStringWithoutBrackets(list1):
    return str(list1).replace('[','').replace(']','')    
#%%
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
#%%
def Neighbor_Check_X2X(X, train_indices, comp, option):
    if option == "PCA":
        X_temp = PCA(n_components=comp).fit_transform(X)
    if option == "TSNE":
        X_temp = TSNE(n_components=comp).fit_transform(X)
    clf = NearestNeighbors(n_neighbors = 2).fit(X_temp[train_indices])
    neigbours = clf.kneighbors(X_temp[train_indices])[1]
    neigbours = [list(x) for x in neigbours]
    neigbours_labels = [[Y[train_indices[i]] for i in ind] for ind in neigbours]
    corrected_pairs = [1 for i in range(len(train_indices)) if np.min(neigbours_labels[i]) == np.max(neigbours_labels[i])]
    prob = np.sum(corrected_pairs)/len(Y[train_indices])
    return prob
#%%  # dataset: "pollen", "baron", "muraro", "patel", "xin", "zeisel"
for prefixFileName in ["pollen", "patel"]:  
    fileName = "../Data/" + prefixFileName + "-prepare-log_count_100pca.csv"
    # choice = "mean",  choice = L \in [0,1]
    # alg = "srnc", alg = "svm", alg = "dt",  alg = "lr"
    choice = 0  #choice = 1 --> "min", choice = 0 --> "max",
    add = 1
    alg = "LinearSVC"
#    alg = "LinearSVC"
#    data_seed = int(sys.argv[1])
    real_random_number = int(2000000*random.random()) # get real random number for cross validation
    times = 10 
    df = pd.read_csv(fileName)
    XY= df.values
    X= XY[:,1:]
    Y= XY[:,0].astype(int)#
#    X = TSNE(n_components=2).fit_transform(X)
#    k = 2
 #   X = MinMaxScaler().fit_transform(X)
#    X = StandardScaler().fit_transform(X)
    Y_all_labels = list(set(Y))
  #   for data_seed in [see for see in range(10)]: 
    ARI_merge_clusters = []
    ARI_SequentialRadiusNeighborsClassifier = []
    ARI_louvain = []
    for repeat_time in range(times):
#        data_seed += repeat_time
        data_seed = repeat_time
        print("data_seed:", data_seed)
        # print("data_seed: ", data_seed)
        train_indices, test_indices, unknown_classes = SplitData(X, Y, Y_all_labels, random_seed = data_seed)
        option = "PCA"
#        alg = "svm"
        comps_checked_PCA = range(1, np.min((X[train_indices]).shape))
        probs_matched_PCA = [Neighbor_Check_X2X(X, train_indices, comp, option) for comp in comps_checked_PCA]
        opt_comp_ind_PCA = comps_checked_PCA[probs_matched_PCA.index(np.max(probs_matched_PCA))]
        X_PCA = PCA(n_components=opt_comp_ind_PCA).fit_transform(X)
        print(opt_comp_ind_PCA, np.max(probs_matched_PCA))
    # run Louvain algorithm
        # n_clusters = int(len(np.unique(Y_test)))
        # print("n_clusters: ", n_clusters)
        # louvain_labels = louvain_exact_K(X_test, n_clusters)
        # ARI_louvain.append(adjusted_rand_score(louvain_labels, Y_test))
        # print("Louvain done!")
        #Run internal cross validation to choose K and epsilon   
        Y_predict_src = SequentialRadiusNeighborsClassifier(X_PCA, Y_all_labels, X_PCA[train_indices], X_PCA[test_indices], Y[train_indices], add, alg)
        ARI_Srn_repeat_time = adjusted_rand_score(Y_predict_src, Y[test_indices])
        ARI_SequentialRadiusNeighborsClassifier.append(ARI_Srn_repeat_time)
        ## Merge clusters
#        Y_predict_merge_clusters = SequentialRadiusNeighborsClassifier(X_PCA, Y_all_labels, X_PCA[train_indices], X_PCA[test_indices], Y[train_indices], add, alg)
        Y_predict_merge_clusters = Y_predict_src
        print("merge label to match the ground truth")
        Y_predict_merge_clusters = matchLabel(Y_predict_merge_clusters, Y[test_indices])
        ARI_Srn_merge_clusters_repeat_time = adjusted_rand_score(Y_predict_merge_clusters, Y[test_indices])
        ARI_merge_clusters.append(ARI_Srn_merge_clusters_repeat_time)
        known_unknown_test = [0 if Y[i] in unknown_classes else 1 for i in test_indices] 
        accuracy_known_test = np.sum([1 for i in test_indices if known_unknown_test[test_indices.index(i)] ==1 and Y[i] == Y_predict_src[test_indices.index(i)]])/np.sum(known_unknown_test)
        known_classes = [i for i in list(set(Y[test_indices])) if i not in unknown_classes]
        accuracy_unknown_correct_test = np.sum([1 for i in range(len(test_indices)) if Y_predict_src[i] not in list(set(Y[test_indices])) and known_unknown_test[i]==0])/(len(test_indices) -np.sum(known_unknown_test)) 
        print("-------------------------------------------------------------------")
        print(set(Y))
        print(set(Y[train_indices]))
        print(set(Y[test_indices]))
        print(set(Y_predict_src))
        print("number_of_true_class:", len(set(Y[test_indices])))
        print("number_of_cluster_Srn_repeat_time:", len(set(Y_predict_src)))
        print("number_of_cluster_Srn_merge_clusters_repeat_time:", len(set(Y_predict_merge_clusters)))
        print("accuracy_known_test", accuracy_known_test)
        print("accuracy_unknown_correct_test", accuracy_unknown_correct_test)
        print("-------------------------------------------------------------------")
        print("ARI_Srn_repeat_time:", ARI_Srn_repeat_time)
        print("ARI_Srn_merge_clusters_repeat:", ARI_Srn_merge_clusters_repeat_time)
        print("===================================================================")
#        print("ARI_Srn               :", (ARI_SequentialRadiusNeighborsClassifier))
#        print("ARI_Srn_merge_clusters:", ARI_merge_clusters)
#        print("ARI_louvain           :", ARI_louvain)
        #Saving results 
        ids = [i for i in range(X.shape[0])]
        train_1_test_0_ids = [1 if i in train_indices else 0 for i in ids]
        true_labels = [Y[i] for i in ids]
        predicted_labels = [-1 for i in ids]
        predicted_labels = [Y_predict_src[test_indices.index(i)] if i in test_indices else predicted_labels[i] for i in ids]
        known_unknown_test = [0 if Y[i] in unknown_classes else 1 for i in test_indices]
        known_1_unknown_0_classes = [known_unknown_test[test_indices.index(i)] if i in test_indices else predicted_labels[i] for i in ids]
        print("fileName = ", str(prefixFileName), "choice = ", str(choice), "alg = ", str(alg), "add = ", str(add))
        df = pd.DataFrame(data= {'ids': ids, 'train_1_test_0_ids': train_1_test_0_ids, 'true_labels':true_labels, 'predicted_labels': predicted_labels, 'known_1_unknown_0_classes': known_1_unknown_0_classes})
        df.to_csv("output/CV_0_Gamma_0/" +prefixFileName+ "_ARI_dataseed_"+str(data_seed)+"_add_"+str(add)+"_eps_"+str(choice)+"_alg_"+str(alg)+".csv", index=False)
    print("===================================================================")
    print("ARI_Srn               :", (ARI_SequentialRadiusNeighborsClassifier))
    print("ARI_Srn_merge_clusters:", ARI_merge_clusters)
    print("ARI_louvain           :", ARI_louvain)
  #    df_lv = pd.DataFrame(data={'ARI_louvain': ARI_louvain})
  #    df_lv.to_csv("output/CV_0_Gamma_0/" +prefixFileName+ "_ARI_louvain_dataseed_"+str(data_seed)+".csv", index=False)
    

# #from sklearn.datasets import make_classification
# #import matplotlib.pyplot as plt
# #from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import RadiusNeighborsClassifier
# #import math
# import sys
# import numpy as np
# from sklearn.model_selection import train_test_split
# #import operator
# import random
# from random import choices
# from sklearn.metrics.cluster import adjusted_rand_score
# from sklearn.neighbors import NearestNeighbors
# #from scipy.spatial import distance
# import pandas as pd
# from sklearn import svm
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import LinearSVC
# from sklearn import linear_model
# #from scipy.stats import bernoulli
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# from collections import Counter
# from sklearn.manifold import TSNE
# #from eval_bisect_louvain import louvain_exact_K #import louvain clustering 
# #%%
# def MinDistanceX1_2_X2(k, X_test, X):
# #    alpha = 0.05
#     Knn_temp = NearestNeighbors(n_neighbors = k + 1).fit(X)
#     distances = Knn_temp.kneighbors(X_test)[0]
#     neighbors = Knn_temp.kneighbors(X_test)[1]
#     return [[x[-1] for x in distances], [x[-1] for x in neighbors]]
# #    length = extrated_distances.shape[0]
        
# #%%
# def Classifier(new_classes, optimal_test, epsilon_choice, X_train_temp, X_test, Y_train_temp, alg):
#     clf = RadiusNeighborsClassifier(radius=epsilon_choice, weights='distance').fit(X_train_temp, Y_train_temp)        
#     predict_set = clf.radius_neighbors(X_test[optimal_test].reshape(1, -1))[1]
#     predict_set = list(predict_set[0])
#     if len(predict_set)> 1:
#         if min(Y_train_temp[predict_set]) == max(Y_train_temp[predict_set]):
#                  return  [min(Y_train_temp[predict_set]), predict_set] 
#         else:
#             if alg == "srnc":
#                y_predict = clf.predict(X_test[optimal_test].reshape(1, -1))        
#             else:                        
#                 if alg == "svm":
#                     clf = svm.SVC().fit(X_train_temp[predict_set], Y_train_temp[predict_set])
#                 if alg == "LinearSVC":
#                     clf = LinearSVC(max_iter=1000).fit(X_train_temp[predict_set], Y_train_temp[predict_set])
#                 if alg == "sgd":
#                     clf = linear_model.SGDClassifier().fit(X_train_temp[predict_set], Y_train_temp[predict_set])
#                 if alg == "dt":
#                     clf = DecisionTreeClassifier().fit(X_train_temp[predict_set], Y_train_temp[predict_set])
#                 if alg == "rf":
#                     clf = RandomForestClassifier(n_estimators=10).fit(X_train_temp[predict_set], Y_train_temp[predict_set])
#                 if alg == "gb":
#                     clf = GradientBoostingClassifier(n_estimators=10).fit(X_train_temp[predict_set], Y_train_temp[predict_set])
#                 if alg == "lr":
#                     clf = LogisticRegression(max_iter=1000).fit(X_train_temp[predict_set], Y_train_temp[predict_set])
#                 if alg == "mlp":
#                     clf = MLPClassifier().fit(X_train_temp[predict_set], Y_train_temp[predict_set])
#                 y_predict = clf.predict(X_test[optimal_test].reshape(1, -1))
#             return [y_predict[0], predict_set]
#     else:
#         return [new_classes, predict_set]  
# #%% Algorithm 1 
# def SequentialRadiusNeighborsClassifier(X, Y, Y_test, Y_all_labels, X_train, X_test, Y_train, add, alg):
# #    size_train = len(Y_train)
#     X_train_temp =  np.copy(X_train)
#     Y_train_temp =  np.copy(Y_train)
#     test_size = len(X_test)
#     train_size = len(X_train)
#     Y_predict = [-1 for x in range(test_size)]
#     test_index = [x for x in range(test_size)]
#     new_indices = []
#     Y_train_all_labels = list(set(Y_train))
#     new_classes = len(Y_all_labels) + 1 
#     distances_X_test_2_X , neighbors_X_test_2_X = MinDistanceX1_2_X2(1, X_test, X)
#     print([[Y_test[te], Y[int(neighbors_X_test_2_X[te])]] for te in range(test_size)])
#     sys.exit()
#     indices, L_sorted = zip(*sorted(enumerate(MinDistanceX_test_2_X), key=itemgetter(1)))
# #    print(epsilon_set)
# #    sys.exit()
# #    epsilon_linspace = np.linspace(epsilon_set[0], epsilon_set[1], 10)
# #    epsilon_update = updateEpsilon(distances, test_index, choice)
#     for test_time in range(test_size):
# #        epsilon_set = [LearnEpsilon(1, X_train_temp, 1), LearnEpsilon(1, X_train_temp, 0)]
# #        epsilon_linspace = np.linspace(epsilon_set[0], epsilon_set[1], 10)
#         Knn_temp = NearestNeighbors(n_neighbors=1)
#         Knn_temp.fit(X_train)
#         min_distances = Knn_temp.kneighbors(X_test[test_index])[0]
#         min_distances = [x for x in min_distances]
#         optimal_indice = min_distances.index(np.min(min_distances))
#         optimal_test = test_index[optimal_indice]
# #        y_predict_linspace = Classifier(new_classes, optimal_test, epsilon_choice, X_train_temp, X_test, Y_train_temp, alg) 
#         y_predict, predict_set = Classifier(new_classes, optimal_test, epsilon_choice, X_train_temp, X_test, Y_train_temp, alg) 
# #        y_predict = EnsembleClassifier(train_size, new_classes, optimal_test, Y_train_all_labels, epsilon_linspace, X_train_temp, X_test, Y_train_temp, alg)
# #            print(num_y_predict)
# #        sys.exit()
# #        sys.exit()
# #        print(y_predict_linspace)
#         if y_predict == new_classes:
#             X_train_temp = np.append(X_train_temp, [X_test[optimal_test]], axis =0)
#             Y_train_temp = np.append(Y_train_temp, [y_predict], axis =0)
# #            print(np.where(Y_train_temp == new_classes)[0])
#             new_classes += 1
#             new_indices.append(optimal_test)
#         else:
#             if add == 1:
# #            if num_y_predict/len(y_predict_linspace) > 0.8:
#                 X_train_temp = np.append(X_train_temp, [X_test[optimal_test]], axis =0)
#                 Y_train_temp = np.append(Y_train_temp, [y_predict], axis =0)
#                 new_indices.append(optimal_test)
# #                epsilon_linspace = np.append(epsilon_linspace, [np.min(min_distances)], axis =0)
# #            epsilon_update = updateEpsilon(distances, test_index, choice)
#         Y_predict[optimal_test] = y_predict 
#         test_index.remove(optimal_test)
# #    sys.exit()
#     return Y_predict
# #%% 
# def SplitData(X, Y, Y_all_labels, random_seed=-1):
#     if random_seed != -1:
#         print("generate train and test data with seed number: ", random_seed)
#         random.seed(random_seed) # use user random seed
#     unknown_classes = list(set(choices(Y_all_labels, k=int(len(Y_all_labels)/2))))
#     print(unknown_classes)
#     ids = np.array([i for i in range(X.shape[0])])
#     train_indices, test_indices = train_test_split(ids, test_size = 0.1, random_state = 0)
#     leave_out_indices = [i for i in train_indices if Y[i] in unknown_classes]
#     train_indices = [i for i in train_indices if i not in leave_out_indices]
#     test_indices = [i for i in ids if i not in train_indices]
# #    sys.exit()
#     return [train_indices, test_indices, unknown_classes]
# #%%    
# #%%
# def listToStringWithoutBrackets(list1):
#     return str(list1).replace('[','').replace(']','')    

# def matchLabel(Y_labels, Y_ref):
#     # this function changes clusters' label of Y_labels s.t ARI(Y_labels, Y_ref) is maximal.
#     Y_labels_set = np.unique(Y_labels)
#     Y_ref_set = np.unique(Y_ref)
#     Y_result = np.copy(Y_labels)
#     for x in Y_labels_set:
#         arg_index = -1
#         max_value = -1
#         for y in Y_ref_set:
#             setA = (Y_labels == x)
#             setB = (Y_ref == y)
#             sumAB = np.sum(setA*setB)
#             if sumAB > max_value:
#                 max_value = sumAB 
#                 arg_index = y
#         Y_result[Y_labels==x] = arg_index #replace x in Y_labels by arg_index
#     return Y_result

# #%%  # dataset: "pollen", "baron", "muraro", "patel", "xin", "zeisel"
# for prefixFileName in ["pollen"]:  
#     fileName = "../Data/" + prefixFileName + "-prepare-log_count_100pca.csv"
#     # choice = "mean",  choice = L \in [0,1]
#     # alg = "srnc", alg = "svm", alg = "dt",  alg = "lr"
#     choice = 0  #choice = 1 --> "min", choice = 0 --> "max",
#     add = 1
#     alg = "LinearSVC"
# #    alg = "LinearSVC"
# #    data_seed = int(sys.argv[1])
#     real_random_number = int(1000000*random.random()) # get real random number for cross validation
#     times = 10 
#     df = pd.read_csv(fileName)
#     XY= df.values
#     X= XY[:,1:]
#     Y= XY[:,0].astype(int)
#     X = TSNE(n_components=2).fit_transform(X)
# #    k = 2
# #    X = MinMaxScaler().fit_transform(X)
# #    X = MinMaxScaler().fit_transform(X)
#     Y_all_labels = list(set(Y))
#     #1 - len(Y_all_labels)/(len(Y_all_labels)+1)
# #    epsilon_set = [LearnEpsilon(1, X, "mean"), LearnEpsilon(1, X, 0)]
#  #   for data_seed in [see for see in range(10)]: 
#     ARI_merge_clusters = []
#     ARI_SequentialRadiusNeighborsClassifier = []
#     ARI_louvain = []
#     for repeat_time in range(times):
# #        data_seed += repeat_time
#         data_seed = repeat_time
#         print("data_seed:", data_seed)
#         # print("data_seed: ", data_seed)
#         train_indices, test_indices, unknown_classes = SplitData(X, Y, Y_all_labels, random_seed=data_seed)
#         # run Louvain algorithm
#         # n_clusters = int(len(np.unique(Y_test)))
#         # print("n_clusters: ", n_clusters)
#         # louvain_labels = louvain_exact_K(X_test, n_clusters)
#         # ARI_louvain.append(adjusted_rand_score(louvain_labels, Y_test))
#         # print("Louvain done!")
#         #Run internal cross validation to choose K and epsilon   
#         Y_predict_src = SequentialRadiusNeighborsClassifier(X, Y, Y[test_indices], Y_all_labels, X[train_indices], X[test_indices], Y[train_indices], add, alg)
#         ARI_Srn_repeat_time = adjusted_rand_score(Y_predict_src, Y[test_indices])
#         ARI_SequentialRadiusNeighborsClassifier.append(ARI_Srn_repeat_time)
#         ## Merge clusters
#         Y_predict_merge_clusters = SequentialRadiusNeighborsClassifier(X, Y_all_labels, X[train_indices], X[test_indices], Y[train_indices], add, alg)
#         print("merge label to match the ground truth")
#         Y_predict_merge_clusters = matchLabel(Y_predict_merge_clusters, Y[test_indices])
#         ARI_Srn_merge_clusters_repeat_time = adjusted_rand_score(Y_predict_merge_clusters, Y[test_indices])
#         ARI_merge_clusters.append(ARI_Srn_merge_clusters_repeat_time)
#         print("-------------------------------------------------------------------")
#         print(set(Y))
#         print(set(Y[train_indices]))
#         print(set(Y[test_indices]))
#         print(set(Y_predict_src))
#         print("number_of_true_class:", len(set(Y[test_indices])))
#         print("number_of_cluster_Srn_repeat_time:", len(set(Y_predict_src)))
#         print("number_of_cluster_Srn_merge_clusters_repeat_time:", len(set(Y_predict_merge_clusters)))
#         print("-------------------------------------------------------------------")
#         print("ARI_Srn_repeat_time:", ARI_Srn_repeat_time)
#         print("ARI_Srn_merge_clusters_repeat:", ARI_Srn_merge_clusters_repeat_time)
#         print("===================================================================")
# #        print("ARI_Srn               :", (ARI_SequentialRadiusNeighborsClassifier))
# #        print("ARI_Srn_merge_clusters:", ARI_merge_clusters)
# #        print("ARI_louvain           :", ARI_louvain)
#         #Saving results 
#         ids = [i for i in range(X.shape[0])]
#         train_1_test_0_ids = [1 if i in train_indices else 0 for i in ids]
#         true_labels = [Y[i] for i in ids]
#         predicted_labels = [-1 for i in ids]
#         predicted_labels = [Y_predict_src[test_indices.index(i)] if i in test_indices else predicted_labels[i] for i in ids]
#         known_unknown_test = [0 if Y[i] in unknown_classes else 1 for i in test_indices]
#         known_1_unknown_0_classes = [known_unknown_test[test_indices.index(i)] if i in test_indices else predicted_labels[i] for i in ids]
#         print("fileName = ", str(prefixFileName), "choice = ", str(choice), "alg = ", str(alg), "add = ", str(add))
#         df = pd.DataFrame(data= {'ids': ids, 'train_1_test_0_ids': train_1_test_0_ids, 'true_labels':true_labels, 'predicted_labels': predicted_labels, 'known_1_unknown_0_classes': known_1_unknown_0_classes})
#         df.to_csv("output/CV_0_Gamma_0/" +prefixFileName+ "_ARI_dataseed_"+str(data_seed)+"_add_"+str(add)+"_eps_"+str(choice)+"_alg_"+str(alg)+".csv", index=False)
#     print("===================================================================")
#     print("ARI_Srn               :", (ARI_SequentialRadiusNeighborsClassifier))
#     print("ARI_Srn_merge_clusters:", ARI_merge_clusters)
#     print("ARI_louvain           :", ARI_louvain)
#  #    df_lv = pd.DataFrame(data={'ARI_louvain': ARI_louvain})
#  #    df_lv.to_csv("output/CV_0_Gamma_0/" +prefixFileName+ "_ARI_louvain_dataseed_"+str(data_seed)+".csv", index=False)
    