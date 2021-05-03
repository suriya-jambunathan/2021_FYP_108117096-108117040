#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:06:51 2021

@author: suriyaprakashjambunathan
"""

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


import pandas as pd
import numpy as np

dataset = pd.read_csv('antenna.csv')

#X
X = dataset.loc[:, dataset.columns != 'vswr']
X = X.loc[:, X.columns != 'gain']
X = X.loc[:, X.columns != 'bandwidth']

#y
vswr =avgfit(list(dataset['vswr']))

for i in range(len(vswr)):
    if vswr[i] >= 1 and vswr[i] < 1.52:
        vswr[i] = '1 to 1.2'
    elif vswr[i] >= 1.2 and vswr[i] < 1.4:
        vswr[i] = '1.2 to 1.4'
    elif vswr[i] >= 1.4 and vswr[i] < 1.6:
        vswr[i] = '1.4 to 1.6'
    elif vswr[i] >= 1.6 and vswr[i] < 1.8:
        vswr[i] = '1.6 to 1.8'
    elif vswr[i] >= 1.8 and vswr[i] < 2:
        vswr[i] = '1.8 to 2'
    elif vswr[i] >= 2 and vswr[i] < 2.2:
        vswr[i] = '2 to 2.2'
    elif vswr[i] >= 2.2 and vswr[i] < 2.4:
        vswr[i] = '2.2 to 2.4'
    elif vswr[i] >= 2.4 and vswr[i] < 2.6:
        vswr[i] = '2.4 to 2.6'
    elif vswr[i] >= 2.6 and vswr[i] < 2.8:
        vswr[i] = '2.6 to 2.8'
    elif vswr[i] >= 2.8 and vswr[i] < 3:
        vswr[i] = '2.8 to 3'
    elif vswr[i] >= 3 and vswr[i] < 3.2:
        vswr[i] = '3 to 3.2'
    elif vswr[i] >= 3.2 and vswr[i] < 3.4:
        vswr[i] = '3.2 to 3.4'
    elif vswr[i] >= 3.4 and vswr[i] < 3.6:
        vswr[i] = '3.4 to 3.6'
    elif vswr[i] >= 3.6 and vswr[i] < 3.8:
        vswr[i] = '3.6 to 3.8'
    elif vswr[i] >= 3.8 and vswr[i] < 4:
        vswr[i] = '3.8 to 4'
    elif vswr[i] >= 4:
        vswr[i] = 'greater than 4'
        
        
gain =avgfit(list(dataset['gain']))

for i in range(len(gain)):
    if gain[i] >= 1 and gain[i] < 1.5:
        gain[i] = '1 to 1.5'
    elif gain[i] >= 1.5 and gain[i] < 2:
        gain[i] = '1.5 to 2'
    elif gain[i] >= 2 and gain[i] < 2.5:
        gain[i] = '2 to 2.5'
    elif gain[i] >= 2.5 and gain[i] < 3:
        gain[i] = '2.5 to 3'
    elif gain[i] >= 3 and gain[i] < 3.5:
        gain[i] = '3 to 3.5'
    elif gain[i] >= 3.5 and gain[i] < 4:
        gain[i] = '3.5 to 4'
    elif gain[i] >= 4:
        gain[i] = 'greater than 4'
    elif gain[i] <= 0:
        gain[i] = 'less than 0'


bw = avgfit(list(dataset['bandwidth']))

for i in range(len(bw)):
    if bw[i] >= 60 and bw[i] < 80:
        bw[i] = '60 to 80'
    elif bw[i] >= 80 and bw[i] < 100:
        bw[i] = '80 to 100'
    elif bw[i] >= 100 and bw[i] < 110:
        bw[i] = '100 to 110'
    elif bw[i] >= 110 and bw[i] < 120:
        bw[i] = '110 to 120'
    elif bw[i] >= 120 :
        bw[i] = 'greater than 120'
    elif bw[i] < 60:
        bw[i] = 'less than 60'
'''
vswr =avgfit(list(dataset['vswr']))
gain =avgfit(list(dataset['gain']))
bw = avgfit(list(dataset['bandwidth']))

'''


y1 = pd.DataFrame(bw)
y2 = pd.DataFrame(gain)
y3 = pd.DataFrame(vswr)


X = dataset.loc[:, dataset.columns != 'vswr']
X = X.loc[:, X.columns != 'gain']
X = X.loc[:, X.columns != 'bandwidth']
cols = X.columns.tolist()

acc_list = []

for i in range(10):
    Xi = X.iloc[:, i]
    Xi = pd.DataFrame(Xi)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y1_train, y1_test = train_test_split(Xi, y1, test_size = 0.3, random_state = 0) 
    X_train, X_test, y2_train, y2_test = train_test_split(Xi, y2, test_size = 0.3, random_state = 0) 
    X_train, X_test, y3_train, y3_test = train_test_split(Xi, y3, test_size = 0.3, random_state = 0) 
    
    # Fitting SVM to the Training set
    from sklearn.svm import SVC
    classifier1 = SVC(kernel = 'linear', random_state = 0)
    classifier1.fit(X_train, y1_train)
    
    # Predicting the Test set results
    y1_pred = classifier1.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm1 = confusion_matrix(y1_test, y1_pred)
    acc1 = accuracy_score(y1_test,y1_pred)
    
    
    classifier2 = SVC(kernel = 'linear', random_state = 0)
    classifier2.fit(X_train, y2_train)
    
    # Predicting the Test set results
    y2_pred = classifier2.predict(X_test)
    
    # Making the Confusion Matrix
    cm2 = confusion_matrix(y2_test, y2_pred)
    acc2 = accuracy_score(y2_test,y2_pred)
    
    
    classifier3 = SVC(kernel = 'linear', random_state = 0)
    classifier3.fit(X_train, y3_train)
    
    # Predicting the Test set results
    y3_pred = classifier3.predict(X_test)
    
    # Making the Confusion Matrix
    cm3 = confusion_matrix(y3_test, y3_pred)
    acc3 = accuracy_score(y3_test,y3_pred)
    
    temp_acc = []
    temp_acc.append(cols[i])
    accu = []
    accu.append(acc1*100)
    accu.append(acc2*100)
    accu.append(acc3*100)
    temp_acc.append(accu)
    acc_list.append(temp_acc)





Xi = X.iloc[:, 0:5]
Xi = pd.DataFrame(Xi)

from sklearn.model_selection import train_test_split
X_train, X_test, y1_train, y1_test = train_test_split(Xi, y1, test_size = 0.3, random_state = 0) 
X_train, X_test, y2_train, y2_test = train_test_split(Xi, y2, test_size = 0.3, random_state = 0) 
X_train, X_test, y3_train, y3_test = train_test_split(Xi, y3, test_size = 0.3, random_state = 0) 

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier1 = SVC(kernel = 'linear', random_state = 0)
classifier1.fit(X_train, y1_train)

# Predicting the Test set results
y1_pred = classifier1.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm1 = confusion_matrix(y1_test, y1_pred)
acc1 = accuracy_score(y1_test,y1_pred)


classifier2 = SVC(kernel = 'linear', random_state = 0)
classifier2.fit(X_train, y2_train)

# Predicting the Test set results
y2_pred = classifier2.predict(X_test)

# Making the Confusion Matrix
cm2 = confusion_matrix(y2_test, y2_pred)
acc2 = accuracy_score(y2_test,y2_pred)


classifier3 = SVC(kernel = 'linear', random_state = 0)
classifier3.fit(X_train, y3_train)

# Predicting the Test set results
y3_pred = classifier3.predict(X_test)

# Making the Confusion Matrix
cm3 = confusion_matrix(y3_test, y3_pred)
acc3 = accuracy_score(y3_test,y3_pred)





l = list(y3_pred)



'''
Wm
Width and height of SRR cell


W0m
Gap between rings

dm
Distance between rings

tm
Width of the rings

rows
Number of SRR cells in a array

Xa
Distance between antenna patch and array

Ya
Distacen between SRR cells in the array

gain
Antenna gain

vswr
Voltage Standing Wave Ratio of the antenna

bandwidth
Bandwidth of the antenna

s
Return Loss (S11) of the antenna

pr
Power radiated by antenna

p0
Power accepted by antenna
'''
