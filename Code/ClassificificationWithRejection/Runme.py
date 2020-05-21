# -run- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:15:01 2020

@author: nguye
"""
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
import Comparison
import sys

import matplotlib
import matplotlib.pyplot as plt

def plot_figure(fig_name, ARI_overall_srnc, accuracy_srnc, recall_unknown_srnc,
            precision_unknown_srnc, F1_unknown_srnc, 
            ARI_overall_rejection, accuracy_rejection, recall_unknown_rejection, 
            precision_unknown_rejection, F1_unknown_rejection,
            ARI_overall_semi, accuracy_semi, recall_unknown_semi, 
            precision_unknown_semi, F1_unknown_semi, split_information_all):
    index = [str(i)+":"+str(split_information_all[i][0])+
             "/"+str(split_information_all[i][1])+"/"+str(split_information_all[i][2])+
             ":"+str(split_information_all[i][3])+"/"+str(split_information_all[i][4]) for i in range(len(ARI_overall_srnc))]
    raw_data = {
#            'ARI_srnc':                     ARI_srnc,
            'accuracy_sc (left)':                accuracy_srnc,
#            'precision_unknown_srnc':       precision_unknown_srnc,
            'F1_sc (left)':              F1_unknown_srnc,
#            'ARI_rejection':                ARI_rejection,
            'accuracy_reject (midle)':           accuracy_rejection,
#            'precision_unknown_rejection':  precision_unknown_rejection,
            'F1_reject (midle)':         F1_unknown_rejection,
            #            'ARI_rejection':                ARI_rejection,
            'accuracy_semi (right)':           accuracy_semi,
#            'precision_unknown_rejection':  precision_unknown_semi,
            'F1_semi (right)':         F1_unknown_semi
        }
#    index = [str(i) for i in range(len(ARI_srnc))]
    df = pd.DataFrame(raw_data, index=index)
#    ax = df.plot.bar(rot=0, color=['r', 'b', 'g', 'm', 'y' ,'r',  'b', 'g', 'm', 'y'], align='center', width=0.8, figsize=(16.0, 10.0))
    ax = df.plot.bar(rot=0, color=['r', 'b', 'r',  'b', 'r',  'b'], align='center', width=0.8, figsize=(16.0, 10.0))
    ax.autoscale(enable=True, axis='y', tight=True)
#    ax.autoscale(tight=True)
    ax.axis()
    plt.ylim(0,1.1)
    plt.gca().legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.savefig("figures/"+str(fig_name)+".png", bbox_inches='tight')
    plt.show()
#%%
def harmonic_mean(score_1, score_2):
    if np.min([score_1, score_2]) == 0:
        return 0.0
    else:
        return 2*(score_1*score_2)/(score_1 + score_2)
#%%
def test_information(train_1_test_0_ids, true_labels, known_1_unknown_0_classes, predicted_labels):
    data_size = len(train_1_test_0_ids)
    max_labels = max(true_labels)
    test_indices = [i for i in range(data_size) if train_1_test_0_ids[i] ==0]
    test_known_indices = [i for i in test_indices if known_1_unknown_0_classes[i] == 1]
    predicted_labels_test = [predicted_labels[i] for i in test_indices]
    true_labels_test = [true_labels[i] for i in test_indices]
    ARI_overall = adjusted_rand_score(true_labels_test, predicted_labels_test)
#    print("ARI_overall:", ARI_overall)
    test_unknown_indices = [i for i in test_indices if known_1_unknown_0_classes[i] == 0]
    if len(test_unknown_indices) > 0:
        correct_test_unknown_indices = [i for i in test_unknown_indices if predicted_labels[i] > max_labels]
        recall_unknown = len(correct_test_unknown_indices)/len(test_unknown_indices)
    else:
        recall_unknown  = 0.0
#    print("recall_unknown:", recall_unknown)
    predict_unknown_indices = [i for i in test_indices if predicted_labels[i] > max_labels]
    if len(predict_unknown_indices) > 0:
        precision_unknown = np.sum([1 for i in predict_unknown_indices if i in test_unknown_indices])/len(predict_unknown_indices)
    else:
       precision_unknown = 0.0 
    print("precision_unknown:", precision_unknown)
    F1_unknown = harmonic_mean(recall_unknown, precision_unknown)
    print("F1_unknown:", F1_unknown)
#    unknow_corect_indices = [i for i in predict_unknown_indices if i in test_unknown_indices]
#    if len(unknow_corect_indices) > 0:
#        ARI_unknown = adjusted_rand_score([predicted_labels[i] for i in unknow_corect_indices], [true_labels[i] for i in unknow_corect_indices])
#    else:
#        ARI_unknown = 0.0            
#    print("ARI_unknown:", ARI_unknown)
    if len(test_known_indices) > 0:
        accuracy = np.sum([1 for i in test_indices if predicted_labels[i] == true_labels[i]])/len(test_indices)
    else:
        accuracy = 0.0
    accuracy +=  np.sum([1 for i in predict_unknown_indices if i in test_unknown_indices])/len(test_indices)
    print("accuracy:", accuracy)
    return[ARI_overall, accuracy, recall_unknown, precision_unknown, F1_unknown]
#%%  
# dataset: "pollen", "baron", "muraro", "patel", "xin", "zeisel"
# speed = [0.1, 17.5, 40, 48, 52, 69, 88]
# lifespan = [2, 8, 70, 1.5, 25, 12, 28]
# lifespan_1 = [2, 3, 4, 3, 5, 1, 8]

# index = [str(i) for i in range(len(speed))]
# df = pd.DataFrame({'speed': speed,  'lifespan': lifespan, 'lifespan_1':lifespan_1}, index=index)
# ax = df.plot.bar(rot=0)
# sys.exit()
# for prefixFileName in ["pollen", "patel", "baron","muraro", "xin", "zeisel"]:
# for prefixFileName in ["pollen", "patel", "baron"]:
for prefixFileName in ["pollen"]:
    for left_out_proportion in [0.0, 0.2, 0.5, 0.8]:
    #    prefixFileName = "pollen"
        ARI_overall_srnc_all = []
        accuracy_srnc_all = []
        recall_unknown_srnc_all = []
        precision_unknown_srnc_all = []
        F1_unknown_srnc_all = []
        ARI_overall_rejection_all = []
        accuracy_rejection_all = []
        recall_unknown_rejection_all = []
        precision_unknown_rejection_all = []
        F1_unknown_rejection_all = []
        split_information_all = []
        ARI_overall_semi_all = []
        accuracy_semi_all = []
        recall_unknown_semi_all = []
        precision_unknown_semi_all = []
        F1_unknown_semi_all = []
        #for predictive_alg in ["svm", "lr", "LinearSVM"]:
        for data_seed in range(10):
            # predictive_alg: src", "svm", "dt",  alg = "lr", "sgd", "gb", "mlp", "LinearSVM", "lda"
            predictive_alg = "SVM"
            # embedded_option: "PCA"
            embedded_option = "PCA"
            # control_neighbor: 1, 2, .....
            control_neighbor = 1
            # shrink_parameter = 1, 0.95, 0.9, ...
            shrink_parameter = 1
            # threshold_rejection = 1, 0.9, 0.8, ...
            threshold_rejection = 0.7
            # max_proportion_unknown = 1, 0.9, 0.8, ...
            proportion_unknown = 0.4
            # max_proportion_unknown = 1, 0.9, 0.8, ...
        #    left_out_proportion = 0.0
            # filter_proportion = 5, 10, 15
            filter_proportion = 0
            methods = ["srnc", "reje", "semi"]
            methods = ["semi"]
            Comparison.main(prefixFileName, data_seed, predictive_alg, embedded_option, control_neighbor, 
                            shrink_parameter, threshold_rejection, proportion_unknown, left_out_proportion, filter_proportion, methods)
            # compute results
            #load results
            results_file_name = "results/"+prefixFileName+"-srnc-dataseed-"+str(data_seed)+"-predictive_alg-"+str(predictive_alg)+"-embedded_option-"+str(embedded_option)+"-shrink_parameter-"+str(shrink_parameter)+"-left_out_proportion-"+str(left_out_proportion)+".csv"
            df = pd.read_csv(results_file_name)
            train_1_test_0_ids = list(df.loc[:,"train_1_test_0_ids"])
            true_labels	= list(df.loc[:,"true_labels"])
            predicted_labels_srnc = list(df.loc[:,"predicted_labels_srnc"])
            known_1_unknown_0_classes = list(df.loc[:,"known_1_unknown_0_classes"])	
            #load results
            results_file_name = "results/"+prefixFileName+"-reje-dataseed-"+str(data_seed)+"-predictive_alg-"+str(predictive_alg)+"-embedded_option-"+str(embedded_option)+"-shrink_parameter-"+str(shrink_parameter)+"-left_out_proportion-"+str(left_out_proportion)+".csv"
            df = pd.read_csv(results_file_name) 
            predicted_labels_rejection = list(df.loc[:,"predicted_labels_rejection"])
            #load results
            results_file_name = "results/"+prefixFileName+"-semi-dataseed-"+str(data_seed)+"-predictive_alg-"+str(predictive_alg)+"-embedded_option-"+str(embedded_option)+"-shrink_parameter-"+str(shrink_parameter)+"-left_out_proportion-"+str(left_out_proportion)+".csv"
            df = pd.read_csv(results_file_name) 
            predicted_labels_semi = list(df.loc[:,"predicted_labels_semi"])
        #    predicted_labels_spectral = list(df.loc[:,"predicted_labels_spectral"])
            print("-------------------------------------------------------------------")
            print("predicted_labels_srnc")
            ARI_overall_srnc_fold, accuracy_srnc_fold, recall_unknown_srnc_fold, precision_unknown_srnc_fold, F1_unknown_srnc_fold = test_information(train_1_test_0_ids, true_labels, known_1_unknown_0_classes, predicted_labels_srnc)
            ARI_overall_srnc_all.append(ARI_overall_srnc_fold)
            accuracy_srnc_all.append(accuracy_srnc_fold)
            recall_unknown_srnc_all.append(recall_unknown_srnc_fold)
            precision_unknown_srnc_all.append(precision_unknown_srnc_fold)
            F1_unknown_srnc_all.append(F1_unknown_srnc_fold)
            print("-------------------------------------------------------------------")
            print("predicted_labels_rejection")
            ARI_overall_rejection_fold, accuracy_rejection_fold, recall_unknown_rejection_fold, precision_unknown_rejection_fold, F1_unknown_rejection_fold = test_information(train_1_test_0_ids, true_labels, known_1_unknown_0_classes, predicted_labels_rejection)
            ARI_overall_rejection_all.append(ARI_overall_rejection_fold)
            accuracy_rejection_all.append(accuracy_rejection_fold)
            recall_unknown_rejection_all.append(recall_unknown_rejection_fold)
            precision_unknown_rejection_all.append(precision_unknown_rejection_fold)
            F1_unknown_rejection_all.append(F1_unknown_rejection_fold)
            print("-------------------------------------------------------------------")
            print("predicted_labels_semi")
            ARI_overall_semi_fold, accuracy_semi_fold, recall_unknown_semi_fold, precision_unknown_semi_fold, F1_unknown_semi_fold = test_information(train_1_test_0_ids, true_labels, known_1_unknown_0_classes, predicted_labels_semi)
            ARI_overall_semi_all.append(ARI_overall_semi_fold)
            accuracy_semi_all.append(accuracy_semi_fold)
            recall_unknown_semi_all.append(recall_unknown_semi_fold)
            precision_unknown_semi_all.append(precision_unknown_semi_fold)
            F1_unknown_semi_all.append(F1_unknown_semi_fold)            
            print("-------------------------------------------------------------------")            
            data_size = len(known_1_unknown_0_classes)
            number_unknown_classe = np.unique([true_labels[i] for i in range(data_size) if known_1_unknown_0_classes[i] == 0]).shape[0]
            split_information = [known_1_unknown_0_classes.count(-1)/data_size, 
                                 known_1_unknown_0_classes.count(1)/data_size, 
                                 known_1_unknown_0_classes.count(0)/data_size]
            split_information = [int(round(x*100, 1)) for x in split_information] + [int(round(number_unknown_classe, 1)), int(round(np.unique(true_labels).shape[0], 1))]
            split_information_all.append(split_information)
            # test_indices = [i for i in range(len(train_1_test_0_ids)) if train_1_test_0_ids[i] ==0]
            # predicted_labels_test_spectral = [predicted_labels_spectral[i] for i in test_indices]
            # true_labels_test_spectral = [true_labels[i] for i in test_indices]
            # ARI_overall_spectral = adjusted_rand_score(true_labels_test_spectral, predicted_labels_test_spectral)
            # ARI_overall_spectral_all.append(ARI_overall_spectral_all)
            print("===================================================================")
        fig_name = prefixFileName+"-ARI-"+"-predictive_alg-"+str(predictive_alg)+"-embedded_option-"+str(embedded_option)+"-control_neighbor-"+str(control_neighbor)+"-left_out_proportion-"+str(int(left_out_proportion*10))
        plot_figure(fig_name, ARI_overall_srnc_all, accuracy_srnc_all, recall_unknown_srnc_all,
                    precision_unknown_srnc_all, F1_unknown_srnc_all, 
                    ARI_overall_rejection_all, accuracy_rejection_all, recall_unknown_rejection_all, 
                    precision_unknown_rejection_all, F1_unknown_rejection_all, 
                    ARI_overall_semi_all, accuracy_semi_all, recall_unknown_semi_all, 
                    precision_unknown_semi_all, F1_unknown_semi_all, split_information_all)
        
        # Note: run smaller shrink_parameter == 1
        # Note: run smaller threshold for rejection == 1
