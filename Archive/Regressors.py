#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:02:07 2021

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

#Regressors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import LassoLarsIC
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

# Defining the Regressor


# Bandwidth

function_r = []

for i in ['"ls"', '"lad"', '"huber"', '"quantile"']:
    for j in range(1,10,2):
        for k in range(10,151,20):
            for l in ['"friedman_mse"', '"mse"', '"mae"']:
                function_r.append("GradientBoostingRegressor(loss =" + i + ", learning_rate =" + str(j/10) + ", n_estimators =" + str(k) + ", criterion =" + l + ", random_state = 0)")
                  
for i in range(1,201):
    for j in ['"mse"', '"mae"']:
        function_r.append("RandomForestRegressor(n_estimators =" + str(i) + ",criterion =" + j + ", random_state = 0)")

for i in range(10,111):
    function_r.append("BaggingRegressor(n_estimators =" + str(i) + ")")
    
for i in range(5,151,5):
    for j in ['"mse"', '"mae"']:    
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_r.append("ExtraTreesRegressor(n_estimators =" + str(i) + ", criterion =" + j + ", max_features =" + k + ", random_state = 0)")
            
for i in ['"mse"', '"friedman_mse"', '"mae"', '"poisson"']:
    for j in ['"best"', '"random"']:
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_r.append("DecisionTreeRegressor(criterion =" + i + ", splitter =" + j + ", max_features =" + k + ", random_state = 0)")


bw_func_r = function_r

del function_r


# Gain

function_r = []

for i in range(5,151,1):
    for j in ['"mse"', '"mae"']:    
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_r.append("ExtraTreesRegressor(n_estimators =" + str(i) + ", criterion =" + j + ", max_features =" + k + ", random_state = 0)")
      
for i in ['"mse"', '"friedman_mse"', '"mae"', '"poisson"']:
    for j in ['"best"', '"random"']:
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_r.append("DecisionTreeRegressor(criterion =" + i + ", splitter =" + j + ", max_features =" + k + ", random_state = 0)")

for i in ['"ls"', '"lad"', '"huber"', '"quantile"']:
    for j in range(1,20):
        for k in range(10,151,20):
            for l in ['"friedman_mse"', '"mse"', '"mae"']:
                function_r.append("GradientBoostingRegressor(loss =" + i + ", learning_rate =" + str(j/20) + ", n_estimators =" + str(k) + ", criterion =" + l + ", random_state = 0)")
                  
for i in ['"mse"', '"friedman_mse"', '"mae"']:
    for j in ['"best"', '"random"']:
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_r.append("ExtraTreeRegressor(criterion =" + i + ", splitter =" + j + ", max_features =" + k + ", random_state = 0)")

for i in ['"bic"' , '"aic"']:
    for j in range(100,1000,100):
        function_r.append("LassoLarsIC(criterion =" + i + ", max_iter =" + str(j) + ")")

gain_func_r = function_r

del function_r


# VSWR

function_r = []

for i in ['"mse"', '"friedman_mse"', '"mae"', '"poisson"']:
    for j in ['"best"', '"random"']:
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_r.append("DecisionTreeRegressor(criterion =" + i + ", splitter =" + j + ", max_features =" + k + ", random_state = 0)")

for i in ['"ls"', '"lad"', '"huber"', '"quantile"']:
    for j in range(1,10,2):
        for k in range(10,151,20):
            for l in ['"friedman_mse"', '"mse"', '"mae"']:
                function_r.append("GradientBoostingRegressor(loss =" + i + ", learning_rate =" + str(j/10) + ", n_estimators =" + str(k) + ", criterion =" + l + ", random_state = 0)")
             
for i in range(10,111):
    function_r.append("BaggingRegressor(n_estimators =" + str(i) + ")")
    
for i in range(1,201):
    for j in ['"mse"', '"mae"']:
        function_r.append("RandomForestRegressor(n_estimators =" + str(i) + ",criterion =" + j + ", random_state = 0)")

for i in range(5,151,1):
    for j in ['"mse"', '"mae"']:    
        for k in ['"auto"', '"sqrt"', '"log2"']:
            function_r.append("ExtraTreesRegressor(n_estimators =" + str(i) + ", criterion =" + j + ", max_features =" + k + ", random_state = 0)")
      
vswr_func_r = function_r

del function_r




max_acc = []
acc_conf = []

reg_bw = []
reg_gain = []
reg_vswr = []

import time


start = time.time()
laps = []
for param in params:
    print(param)
    function_r = []
    if param == 'bandwidth':
        function_r = bw_func_r
    elif param == 'gain':
        function_r = gain_func_r
    elif param == 'vswr':
        function_r = vswr_func_r
        
    # Splitting into Test and Train set
    X_train, X_test, y_train, y_test = train_test_split(Xi, y[param], test_size = 0.3, random_state = 0) 
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    
    count = [(list(y_train[param])).count(x) for x in list(set(list(y_train[param])))]
    
    class_weights = dict(zip(list(set(list(y_train[param]))),count))
    
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    list_6 = []
    
    # Splitting the train set into specific labels for Regression Training
    
    for i in range(572):
        try:
            if 'Class 1' in y_train[param][i]:
                list_1.append(i)
            elif 'Class 2' in y_train[param][i]:
                list_2.append(i)
            elif 'Class 3' in y_train[param][i]:
                list_3.append(i)
            elif 'Class 4' in y_train[param][i]:
                list_4.append(i)
            elif 'Class 5' in y_train[param][i]:
                list_5.append(i)
            elif 'Class 6' in y_train[param][i]:
                list_6.append(i)
        except:
            continue
    
    X_train_1 = X_train.loc[list_1]
    X_train_2 = X_train.loc[list_2]
    X_train_3 = X_train.loc[list_3]
    X_train_4 = X_train.loc[list_4]
    X_train_5 = X_train.loc[list_5]
    X_train_6 = X_train.loc[list_6]
    
    y_train_1 = (dataset[param]).loc[list_1]
    y_train_2 = (dataset[param]).loc[list_2]
    y_train_3 = (dataset[param]).loc[list_3]
    y_train_4 = (dataset[param]).loc[list_4]
    y_train_5 = (dataset[param]).loc[list_5]
    y_train_6 = (dataset[param]).loc[list_6]
    
    reg = []
    for func_r in function_r:
        try :
            reg_1 = eval(func_r)
            reg_1.fit(X_train_1, y_train_1)
            
            reg_2 = eval(func_r)
            reg_2.fit(X_train_2, y_train_2)
            
            reg_3 = eval(func_r)
            reg_3.fit(X_train_3, y_train_3)
            
            reg_4 = eval(func_r)
            reg_4.fit(X_train_4, y_train_4)
            
            reg_5 = eval(func_r)
            reg_5.fit(X_train_5, y_train_5)
            
            reg_6 = eval(func_r)
            reg_6.fit(X_train_6, y_train_6)
            print(func_r)
            if param == 'bandwidth':
                reg_bw.append([reg_1,reg_2,reg_3,reg_4,reg_5,reg_6])
            elif param == 'gain':
                reg_gain.append([reg_1,reg_2,reg_3,reg_4,reg_5,reg_6])
            elif param == 'vswr':
                reg_vswr.append([reg_1,reg_2,reg_3,reg_4,reg_5,reg_6])
        except:
            continue
    laps.append(time.time())
    
# save
from sklearn.externals import joblib
filename_1 = 'reg_bw.sav'
joblib.dump(reg_bw, filename_1) 
 
filename_2 = 'reg_gain.sav'
joblib.dump(reg_gain, filename_2) 

filename_3 = 'reg_vswr.sav'
joblib.dump(reg_vswr, filename_3) 
