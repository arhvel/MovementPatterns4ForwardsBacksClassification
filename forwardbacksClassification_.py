#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 19:44:55 2022

@author: adeyem01
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
from numpy import mean
from numpy import std
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, accuracy_score, matthews_corrcoef, precision_score, recall_score,f1_score, cohen_kappa_score, roc_auc_score

# Data
data_bin = pd.read_csv("RugbyLeaguePMPatterns_bin_FB.csv")
data_bin_predictors = data_bin.drop('Position',1)
data_bin_predictors = data_bin_predictors.drop('Ath_id',1)
data_bin_predictors = data_bin_predictors.drop('Fixture',1)
data_bin_target = data_bin['Position']

columns = list(data_bin_predictors.columns)


# Data
data_rel = pd.read_csv("RugbyLeaguePMPatterns_rel_FB.csv")
data_rel_predictors = data_rel.drop('Position',1)
data_rel_predictors = data_rel_predictors.drop('Ath_id',1)
data_rel_predictors = data_rel_predictors.drop('Fixture',1)
data_rel_target = data_rel['Position']


            #Classifiers and modelling
LOGREG = LogisticRegression(penalty='l2', solver='lbfgs',  max_iter=5000)
RF = RandomForestClassifier(random_state=1)
NB = GaussianNB()
DT = tree.DecisionTreeClassifier()
MLP = MLPClassifier(random_state = 5, max_iter=3000, early_stopping = True, solver = 'adam', hidden_layer_sizes = (1000,500,250,125,50,25))
KN = KNeighborsClassifier(n_neighbors=5)

models= [DT, NB, RF, LOGREG, MLP, KN]

cv= KFold(n_splits=10, random_state=10, shuffle=True )

#Model building and evaluation
scoring = {'accuracy' : make_scorer(accuracy_score),'Matt-Coef' : make_scorer(matthews_corrcoef), 'Kappa' : make_scorer(cohen_kappa_score),  'precision' : make_scorer(precision_score, average='macro', zero_division = 0),'recall' : make_scorer(recall_score, average='macro', zero_division = 0), 'f1_score' : make_scorer(f1_score, average='macro', zero_division = 0)}



resultsREL = []
print('========= rel =========')
for model in models:
    scores = cross_validate(model, data_rel_predictors, data_rel_target, scoring= scoring, cv=cv, n_jobs=1)  
    accuracy = mean(scores['test_accuracy']*100)
    accuracy_SD = std(scores['test_accuracy'])
    precision = mean(scores['test_precision'])
    recall = mean(scores['test_recall'])
    f1 = mean(scores['test_f1_score'])
    matt = mean(scores['test_Matt-Coef'])
    kapp = mean(scores['test_Kappa'])
    _com = (str(model), accuracy, accuracy_SD, precision, recall,f1,matt, kapp)
    resultsREL.append(_com)

    # Result Table for CLASSIFICATION

resultFrames = []
headings = ['Classifier','Accuracy', 'SD','Precision', 'Recall', 'F1_Score', 'MattCo-ef', 'Kappa']


REL = pd.DataFrame(resultsREL, columns = headings)
REL = round(REL,2)
REL['ColumnValue'] ='Relative Frequency'
resultFrames.append(REL)

resultFrames = pd.concat(resultFrames).reset_index(drop=True)
resultFrames['Accuracy ' + '\u00B1 ' +  'SD'] = resultFrames['Accuracy'].map(str) + ' \u00B1 ' + resultFrames['SD'].map(str)

resultFrames.to_csv('FB_ClassificationResults.csv', index=False)



# Feature Selection and re-classification

FS_Data = pd.read_csv("RugbyLeaguePMPatterns_rel_FB_CFS_BS.csv")
FS_Data_predictors = FS_Data.drop("Position", axis= 1)
FS_Data_target = FS_Data["Position"]
FS_Data_predictors.columns

resultsREL_FS = []
print('========= rel Feature Selection =========')
for model in models:
    scores = cross_validate(model, FS_Data_predictors, FS_Data_target, scoring= scoring, cv=cv, n_jobs=1)  
    accuracy = mean(scores['test_accuracy']*100)
    accuracy_SD = std(scores['test_accuracy'])
    precision = mean(scores['test_precision'])
    recall = mean(scores['test_recall'])
    f1 = mean(scores['test_f1_score'])
    matt = mean(scores['test_Matt-Coef'])
    kapp = mean(scores['test_Kappa'])
    _com = (str(model), accuracy, accuracy_SD, precision, recall,f1,matt, kapp)
    resultsREL_FS.append(_com)
    
    print('Performance results for %s:' %(str(model)))
    print('Accuracy : ',accuracy,'with a standard deviation of ',accuracy_SD, 'Precision: ', precision, ' Recall: ', recall, 'F1-score: ', f1, 'and Matt Co-ef: ', matt)
    print('==============================================================================================================')


 # Result Table for CLASSIFICATION and feature selection
resultFramesFS = []
resultFramesFS.append(REL)

FS = pd.DataFrame(resultsREL_FS, columns = headings)
FS = round(FS,2)
FS['ColumnValue'] ='Feature Selection'
resultFramesFS.append(FS)

resultFramesFS = pd.concat(resultFramesFS).reset_index(drop=True)
resultFramesFS['Accuracy ' + '\u00B1 ' +  'SD'] = resultFramesFS['Accuracy'].map(str) + ' \u00B1 ' + resultFramesFS['SD'].map(str)

resultFramesFS.to_csv('FB_ClassificationResults_FS.csv', index=False)



# Cohen's D EFFECT SIZE

### Cohen's d Effect size Definition

from numpy.random import randn
from numpy.random import seed
from numpy import mean
from numpy import var
from math import sqrt
 
	
# Backs
Backs = FS_Data[FS_Data.Position == 'Back']
b_des = Backs.describe()
Backs_trans = b_des.T.reset_index()
Backs_trans = round(Backs_trans,3)
Backs_trans['Mean' + '\u00B1' + 'SD'] = Backs_trans['mean'].map(str) + ' \u00B1 ' + Backs_trans['std'].map(str)

# Forwards
Forwards = FS_Data[FS_Data.Position == 'Forward']
f_des = Forwards.describe()
Forwards_transposed = f_des.T.reset_index()
Forwards_transposed = round(Forwards_transposed,3)
Forwards_transposed['Mean ' + '\u00B1 ' +  'SD'] = Forwards_transposed['mean'].map(str) + ' \u00B1 ' + Forwards_transposed['std'].map(str)


columns_fs = FS_Data_predictors.columns
columns_fs


#       NON ZEROS
BACKSpatterns = []
for col in columns_fs:
    m = mean(list(set(Backs[col].to_list()))[1:])
    s = std(list(set(Backs[col].to_list()))[1:])
    
    dum = (col, m,s)
    BACKSpatterns.append(dum)
    
BACKSpatternsDF = pd.DataFrame(BACKSpatterns, columns=['Pattern','mean','std'])  
BACKSpatternsDF = round(BACKSpatternsDF,3)
BACKSpatternsDF['Mean ' + '\u00B1 ' +  'SD'] = BACKSpatternsDF['mean'].map(str) + ' \u00B1 ' + BACKSpatternsDF['std'].map(str)

Forwardspatterns = []
for col in columns_fs:
    m = mean(list(set(Forwards[col].to_list()))[1:])
    s = std(list(set(Forwards[col].to_list()))[1:])
    
    dum = (col, m,s)
    Forwardspatterns.append(dum)
    
ForwardspatternsDF = pd.DataFrame(Forwardspatterns, columns=['Pattern','mean','std'])  
ForwardspatternsDF = round(ForwardspatternsDF,3)
ForwardspatternsDF['Mean ' + '\u00B1 ' +  'SD'] = ForwardspatternsDF['mean'].map(str) + ' \u00B1 ' + ForwardspatternsDF['std'].map(str)

def cohend_nonzeros(c1, c2):
    # remove zeros
    d1, d2 = list(set(c1.to_list()))[1:], list(set(c2.to_list()))[1:]
    
	# calculate the size of samples
    n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
	# calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
	# calculate the effect size
    return (u1 - u2) / s



# Effect size of between both positional groups per key variable
ES = []

for col in columns_fs:
    d = cohend_nonzeros(Forwards[col],Backs[col])
    if d>0:
        
        if(d >=0 and d<= 0.1):
            annon = 'trivial'

        elif(d >0.1 and d <= 0.2):
            annon = 'small'

        elif(d > 0.2 and d <= 0.6):
            annon = 'moderate'

        elif(d > 0.6 and d <= 1.2):
             annon = 'large'
             
        elif(d > 1.2 and d <= 2.0):
             annon = 'very large'
        
        elif(d > 2.0 and d <= 4.0):
             annon = 'nearly perfect'
             
        elif(d > 4.0):
             annon = 'perfect'
    else:
        d_ = d * -1
        if(d_ >=0 and d_<= 0.1):
            annon = 'trivial'

        elif(d_ >0.1 and d_ <= 0.2):
            annon = 'small'

        elif(d_ > 0.2 and d_ <= 0.6):
            annon = 'moderate'

        elif(d_ > 0.6 and d_ <= 1.2):
             annon = 'large'
             
        elif(d_ > 1.2 and d_ <= 2.0):
             annon = 'very large'
        
        elif(d_ > 2.0 and d_ <= 4.0):
             annon = 'nearly perfect'
             
        elif(d_ > 4.0):
             annon = 'perfect'
        
        
    collate = (col,d,annon)
    ES.append(collate)

ES_DF = pd.DataFrame(ES, columns=['Pattern','Effect_Size','Remark'])
ES_DF = round(ES_DF,3)
ES_DF['Forwards' + '(Mean' + '\u00B1' +  'SD)'] = ForwardspatternsDF['Mean ' + '\u00B1 ' +  'SD']
ES_DF['Backs' + '(Mean' + '\u00B1' +  'SD)'] = BACKSpatternsDF['Mean ' + '\u00B1 ' +  'SD']
ES_DF.to_csv("EDA_and_effect_size_nonzeros.csv", index = False)
