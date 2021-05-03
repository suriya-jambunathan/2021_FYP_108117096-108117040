#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:59:52 2021

@author: suriyaprakashjambunathan
"""

# Fitting the nan values with the average
def avgfit(l):
    na = pd.isna(l)
    arr = []
    for i in range(len(l)):
        if na[i] == False:
            arr.append(l[i])
    
    avg = sum(arr)/len(arr)
    
    fit_arr = []
    
    for i in range(len(l)):
        if na[i] == False:
            fit_arr.append(l[i])
        elif na[i] == True:
            fit_arr.append(avg)
    
    return(fit_arr)
    
# Weighted Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = list(y_true), list(y_pred)
    l = len(y_true)
    num = 0
    den = 0
    for i in range(l):
        num = num + (abs(y_pred[i] - y_true[i]))
        den = den + y_true[i]
    return abs(num/den) * 100



# Importing the Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import explained_variance_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.simplefilter(action='ignore')
# Importing the Dataset
dataset = pd.read_csv('antenna.csv')

#X
X = dataset.loc[:, dataset.columns != 'vswr']
X = X.loc[:, X.columns != 'gain']
X = X.loc[:, X.columns != 'bandwidth']
Xi = X.iloc[:, :-3]
Xi = pd.DataFrame(Xi)

#y

bw = avgfit(list(dataset['bandwidth']))
dataset['bandwidth'] = bw

for i in range(len(bw)):
    if bw[i] < 100:
        bw[i] = 'Class 1'
    elif bw[i] >= 100 and bw[i] < 115:
        bw[i] = 'Class 2'
    elif bw[i] >= 115 and bw[i] < 120:
        bw[i] = 'Class 3'
    elif bw[i] >= 120 and bw[i] < 121:
        bw[i] = 'Class 4'
    elif bw[i] >= 121 and bw[i] < 122:
        bw[i] = 'Class 5'
    elif bw[i] >= 122 :
        bw[i] = 'Class 6'
    
                
gain =avgfit(list(dataset['gain']))
dataset['gain'] = gain

for i in range(len(gain)):
    if gain[i] < 1.3:
        gain[i] = 'Class 1'
    elif gain[i] >= 1.3 and gain[i] < 1.5:
        gain[i] = 'Class 2'
    elif gain[i] >= 1.5 and gain[i] < 2.4:
        gain[i] = 'Class 3'
    elif gain[i] >= 2.4 and gain[i] < 2.7:
        gain[i] = 'Class 4'
    elif gain[i] >= 2.7 and gain[i] < 2.9:
        gain[i] = 'Class 5'
    elif gain[i] >= 2.9 and gain[i] < 3.5:
        gain[i] = 'Class 6'

vswr =avgfit(list(dataset['vswr']))
dataset['vswr'] = vswr

for i in range(len(vswr)):
    if vswr[i] >= 1 and vswr[i] < 1.16:
        vswr[i] = 'Class 1'
    elif vswr[i] >= 1.16 and vswr[i] < 1.32:
        vswr[i] = 'Class 2'
    elif vswr[i] >= 1.32 and vswr[i] < 1.5:
        vswr[i] = 'Class 3'
    elif vswr[i] >= 1.5 and vswr[i] < 2:
        vswr[i] = 'Class 4'
    elif vswr[i] >= 2 and vswr[i] < 4:
        vswr[i] = 'Class 5'
    elif vswr[i] >= 4:
        vswr[i] = 'Class 6'

y1 = pd.DataFrame(bw)
y2 = pd.DataFrame(gain)
y3 = pd.DataFrame(vswr)

# Accuracy list
acc_list = []

params = ['bandwidth','gain','vswr']

y = pd.DataFrame()
y['bandwidth'] = bw
y['vswr'] = vswr
y['gain'] = gain

classes = ['Class 1','Class 2','Class 3','Class 4','Class 5','Class 6']

# Defining the Classifier and Regressor

#Classifier
function_name_c = "RandomForestClassifier"
attr_c = "(n_estimators = c_loop,random_state=0,class_weight = class_weights)"

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

# Bandwidth 

function_c = []

for i in range(1,151):
    function_c.append("BaggingClassifier(n_estimators =" + str(i) + ", random_state = 0)")
    
for i in ['"gini"', '"entropy"']:
    for j in ['"best"', '"random"']:         
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_c.append("DecisionTreeClassifier(criterion =" + i + ", splitter =" + j + ", max_features =" + k + ", random_state =0, class_weight = class_weights)")

for i in ['"deviance"']:
    for j in range(1,10):
        for k in range(1,26):
            for l in ['"friedman_mse"', '"mse"', '"mae"']:
                for m in ['"auto"', '"sqrt"', '"log2"']:
                    function_c.append("GradientBoostingClassifier(loss =" + i + ", learning_rate =" + str(j/10) + ", n_estimators =" + str(k*5) + ", criterion =" + l + ", max_features =" + m + ", random_state = 0)")
 
for i in range(1,151,5):
    for j in ['"gini"', '"entropy"']:
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_c.append("ExtraTreesClassifier(n_estimators =" + str(i) + ", criterion =" + j + ", max_features =" + k + ", random_state = 0, class_weight = class_weights)")

for i in range(1,151,5):
    for j in ['"gini"', '"entropy"']:
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_c.append("RandomForestClassifier(n_estimators =" + str(i) + ", criterion =" + j + ", max_features =" + k + ", random_state = 0, class_weight = class_weights)")
 
bw_func_c = function_c

del function_c


# Gain

function_c = []

for i in range(1,151):
    for j in ['"gini"', '"entropy"']:
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_c.append("ExtraTreesClassifier(n_estimators =" + str(i) + ", criterion =" + j + ", max_features =" + k + ", random_state = 0, class_weight = class_weights)")

for i in ['"gini"', '"entropy"']:
    for j in ['"best"', '"random"']:         
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_c.append("DecisionTreeClassifier(criterion =" + i + ", splitter =" + j + ", max_features =" + k + ", random_state =0, class_weight = class_weights)")

for i in range(1,151):
    function_c.append("BaggingClassifier(n_estimators =" + str(i) + ", random_state = 0)")
    
for i in ['"auto"', '"binary_crossentropy"', '"categorical_crossentropy"']:
    for j in (1,10,2):
        for k in (10,211,20):
            function_c.append("HistGradientBoostingClassifier(loss =" + i + ", learning_rate =" + str(j/10) + ", max_iter =" + str(k) + ", random_state = 0)")

for i in range(1,151):
    for j in ['"gini"', '"entropy"']:
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_c.append("RandomForestClassifier(n_estimators =" + str(i) + ", criterion =" + j + ", max_features =" + k + ", random_state = 0, class_weight = class_weights)")
 
gain_func_c = function_c

del function_c


# VSWR

function_c = []

for i in ['"gini"', '"entropy"']:
    for j in ['"best"', '"random"']:         
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_c.append("DecisionTreeClassifier(criterion =" + i + ", splitter =" + j + ", max_features =" + k + ", random_state =0, class_weight = class_weights)")

for i in range(1,151,5):
    for j in ['"gini"', '"entropy"']:
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_c.append("RandomForestClassifier(n_estimators =" + str(i) + ", criterion =" + j + ", max_features =" + k + ", random_state = 0, class_weight = class_weights)")
 
for i in range(1,151,5):
    for j in ['"gini"', '"entropy"']:
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_c.append("ExtraTreesClassifier(n_estimators =" + str(i) + ", criterion =" + j + ", max_features =" + k + ", random_state = 0, class_weight = class_weights)")

for i in ['"deviance"']:
    for j in range(1,10):
        for k in range(1,26):
            for l in ['"friedman_mse"', '"mse"', '"mae"']:
                for m in ['"auto"', '"sqrt"', '"log2"']:
                    function_c.append("GradientBoostingClassifier(loss =" + i + ", learning_rate =" + str(j/10) + ", n_estimators =" + str(k*5) + ", criterion =" + l + ", max_features =" + m + ", random_state = 0)")
 
for i in range(1,151):
    function_c.append("BaggingClassifier(n_estimators =" + str(i) + ", random_state = 0)")
    

vswr_func_c = function_c

del function_c
          

max_acc = []
acc_conf = []

class_bw = []
class_gain = []
class_vswr = []

import time

start = time.time()

laps = []
for param in params:
    print(param)
    function_c = []
    if param == 'bandwidth':
        function_c = bw_func_c
    elif param == 'gain':
        function_c = gain_func_c
    elif param == 'vswr':
        function_c = vswr_func_c
    # Splitting into Test and Train set
    X_train, X_test, y_train, y_test = train_test_split(Xi, y[param], test_size = 0.3, random_state = 0) 
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    
    count = [(list(y_train[param])).count(x) for x in list(set(list(y_train[param])))]
    
    class_weights = dict(zip(list(set(list(y_train[param]))),count))
    
     #CLASSIFICATION
    # Fitting Classifier to the Training set
    
    for func_c in function_c:
        try:
            classifier = eval(func_c)
            classifier.fit(X_train,y_train)
            print(func_c)
    
            if param == 'bandwidth':
                class_bw.append([classifier])
            elif param == 'gain':
                class_gain.append([classifier])
            elif param == 'vswr':
                class_vswr.append([classifier])
        except:
            continue
    laps.append(time.time())
    
    
    
# save
import joblib
filename_1 = 'class_bw.sav'
joblib.dump(class_bw, filename_1) 
 
filename_2 = 'class_gain.sav'
joblib.dump(class_gain, filename_2) 

filename_3 = 'class_vswr.sav'
joblib.dump(class_vswr, filename_3) 
