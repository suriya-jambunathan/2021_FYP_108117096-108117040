#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:12:37 2021

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

#Regressor
function_name_r = "RandomForestRegressor"
attr_r = "(n_estimators = r_loop,random_state = 0)"

max_acc = []
acc_conf = []

for param in params:

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
    
    X_train_1 = X_train.ix[list_1]
    X_train_2 = X_train.ix[list_2]
    X_train_3 = X_train.ix[list_3]
    X_train_4 = X_train.ix[list_4]
    X_train_5 = X_train.ix[list_5]
    X_train_6 = X_train.ix[list_6]
    
    y_train_1 = (dataset[param]).ix[list_1]
    y_train_2 = (dataset[param]).ix[list_2]
    y_train_3 = (dataset[param]).ix[list_3]
    y_train_4 = (dataset[param]).ix[list_4]
    y_train_5 = (dataset[param]).ix[list_5]
    y_train_6 = (dataset[param]).ix[list_6]
    
    reg = []
    for r_loop in range(1,151):
            reg_1 = eval(function_name_r + attr_r)
            reg_1.fit(X_train_1, y_train_1)
            
            reg_2 = eval(function_name_r + attr_r)
            reg_2.fit(X_train_2, y_train_2)
            
            reg_3 = eval(function_name_r + attr_r)
            reg_3.fit(X_train_3, y_train_3)
            
            reg_4 = eval(function_name_r + attr_r)
            reg_4.fit(X_train_4, y_train_4)
            
            reg_5 = eval(function_name_r + attr_r)
            reg_5.fit(X_train_5, y_train_5)
            
            reg_6 = eval(function_name_r + attr_r)
            reg_6.fit(X_train_6, y_train_6)
            print(r_loop)
            reg.append([reg_1,reg_2,reg_3,reg_4,reg_5,reg_6])
    
    print(param)
    for c_loop in range(1,151):
        
        #CLASSIFICATION
        # Fitting Classifier to the Training set
        
        classifier = eval(function_name_c + attr_c)
        classifier.fit(X_train, y_train)
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        y_predl = list(y_pred)
        y_pred = pd.DataFrame(y_predl)
        
        # Making the Confusion Matrix
        cm = confusion_matrix(list(y_test[param]), y_predl)
        acc = accuracy_score(list(y_test[param]), y_predl)
        for r_loop in range(1,151):
            
            testlist_1 = []
            testlist_2 = []
            testlist_3 = []
            testlist_4 = []
            testlist_5 = []
            testlist_6 = []
            
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
        
            xtestix = X_test.index.values.tolist()
            y_pred['actual index'] = xtestix
            
            for i in range(172):
                try:
                    if 'Class 1' in y_pred[0][i]:
                        testlist_1.append(y_pred['actual index'][i])
                    elif 'Class 2' in y_pred[0][i]:
                        testlist_2.append(y_pred['actual index'][i])
                    elif 'Class 3' in y_pred[0][i]:
                        testlist_3.append(y_pred['actual index'][i])
                    elif 'Class 4' in y_pred[0][i]:
                        testlist_4.append(y_pred['actual index'][i])
                    elif 'Class 5' in y_pred[0][i]:
                        testlist_5.append(y_pred['actual index'][i])
                    elif 'Class 6' in y_pred[0][i]:
                        testlist_6.append(y_pred['actual index'][i])
                except:
                    continue
                
            X_test_1 = X_test.ix[testlist_1]
            X_test_2 = X_test.ix[testlist_2]
            X_test_3 = X_test.ix[testlist_3]
            X_test_4 = X_test.ix[testlist_4]
            X_test_5 = X_test.ix[testlist_5]
            X_test_6 = X_test.ix[testlist_6]
            
            y_test_1 = (dataset[param]).ix[testlist_1]
            y_test_2 = (dataset[param]).ix[testlist_2]
            y_test_3 = (dataset[param]).ix[testlist_3]
            y_test_4 = (dataset[param]).ix[testlist_4]
            y_test_5 = (dataset[param]).ix[testlist_5]
            y_test_6 = (dataset[param]).ix[testlist_6]
            
            
            # REGRESSION
            y_pred_1 = (reg[r_loop-1][0]).predict(X_test_1)
            y_pred_2 = (reg[r_loop-1][1]).predict(X_test_2)
            y_pred_3 = (reg[r_loop-1][2]).predict(X_test_3)
            y_pred_4 = (reg[r_loop-1][3]).predict(X_test_4)
            y_pred_5 = (reg[r_loop-1][4]).predict(X_test_5)
            y_pred_6 = (reg[r_loop-1][5]).predict(X_test_6)
            
            
            class_reg_1 = mean_absolute_percentage_error(y_test_1, y_pred_1)
            class_reg_2 = mean_absolute_percentage_error(y_test_2, y_pred_2)
            class_reg_3 = mean_absolute_percentage_error(y_test_3, y_pred_3)
            class_reg_4 = mean_absolute_percentage_error(y_test_4, y_pred_4)
            class_reg_5 = mean_absolute_percentage_error(y_test_5, y_pred_5)
            class_reg_6 = mean_absolute_percentage_error(y_test_6, y_pred_6)
            
            wmape = ((class_reg_1*len(testlist_1)) + 
                     (class_reg_2*len(testlist_2)) + 
                     (class_reg_3*len(testlist_3)) + 
                     (class_reg_4*len(testlist_4)) + 
                     (class_reg_5*len(testlist_5)) +
                     (class_reg_6*len(testlist_6)))
            wmape = wmape/172
            #acc_list.append([param,['Accuracy',acc],['MAPE',wmape]])
            acc_conf.append([param,c_loop,r_loop,wmape])
            print(str(c_loop) + ' , ' + str(r_loop))
            
            
np.savetxt("acc.csv",  
           acc_conf, 
           delimiter =", ",  
           fmt ='% s') 




