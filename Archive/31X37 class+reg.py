#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:57:08 2021

@author: suriyaprakashjambunathan
"""

# Classifiers
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm.classes import OneClassSVM
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture

#Regressors
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble.forest import ExtraTreesRegressor
from sklearn.ensemble.bagging import BaggingRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn.gaussian_process.gpr import GaussianProcessRegressor
from  sklearn.isotonic import IsotonicRegression
from sklearn.linear_model.bayes import ARDRegression
from sklearn.linear_model.huber import HuberRegressor
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.passive_aggressive import PassiveAggressiveRegressor 
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.linear_model.theil_sen import TheilSenRegressor
from sklearn.linear_model.ransac import RANSACRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors.regression import KNeighborsRegressor
from sklearn.neighbors.regression import RadiusNeighborsRegressor
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.tree.tree import ExtraTreeRegressor
from sklearn.svm.classes import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lars
from sklearn.linear_model import LarsCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.svm import NuSVR
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.cross_decomposition import PLSCanonical
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR

# Importing the Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import explained_variance_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.simplefilter(action='ignore')


Name_c = ['BaggingClassifier',
         'BernoulliNB',
         'CalibratedClassifierCV',
         'ComplementNB',
         'DecisionTreeClassifier',
         'DummyClassifier',
         'ExtraTreeClassifier',
         'ExtraTreesClassifier',
         'GaussianNB',
         'GaussianProcessClassifier',
         'GradientBoostingClassifier',
         'HistGradientBoostingClassifier',
         'KNeighborsClassifier',
         'LabelPropagation',
         'LabelSpreading',
         'LinearDiscriminantAnalysis',
         'LinearSVC',
         'LogisticRegression',
         'LogisticRegressionCV',
         'MLPClassifier',
         'MultinomialNB',
         'NearestCentroid',
         'PassiveAggressiveClassifier',
         'Perceptron',
         'QuadraticDiscriminantAnalysis',
         'RadiusNeighborsClassifier',
         'RandomForestClassifier',
         'RidgeClassifier',
         'RidgeClassifierCV',
         'SGDClassifier',
         'SVC']


Name_r = [ "RandomForestRegressor",
        "ExtraTreesRegressor",
        "BaggingRegressor",
        "GradientBoostingRegressor",
        "AdaBoostRegressor",
        "GaussianProcessRegressor",
        "ARDRegression",
        "HuberRegressor",
        "LinearRegression",
        "PassiveAggressiveRegressor",
        "SGDRegressor",
        "TheilSenRegressor",
        "KNeighborsRegressor",
        "RadiusNeighborsRegressor",
        "MLPRegressor",
        "DecisionTreeRegressor",
        "ExtraTreeRegressor",
        "SVR",
        "BayesianRidge",
        "CCA",
        "ElasticNet",
        "ElasticNetCV",
        "KernelRidge",
        "Lars",
        "LarsCV",
        "Lasso",
        "LassoCV",
        "LassoLars",
        "LassoLarsIC",
        "LassoLarsCV",
        "NuSVR",
        "OrthogonalMatchingPursuit",
        "OrthogonalMatchingPursuitCV",
        "PLSCanonical",
        "Ridge",
        "RidgeCV",
        "LinearSVR"]


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
    
    y_true, y_pred = list(y_true), list(y_pred)    

# Weighted Mean Absolute Percentage Error
    
    
def mean_absolute_percentage_error(y_true, y_pred):
    l = len(y_true)
    num = 0
    den = 0
    for i in range(l):
        num = num + (abs(y_pred[i] - y_true[i]))
        den = den + y_true[i]
    return abs(num/den) * 100

def classifiers(Name, X_train,y_train):
    clfs = []
    for i in range(len(Name)):
        classifier = globals()[Name[i]]
        print(classifier)
        Classifier = classifier()
        Classifier.fit(X_train, y_train)
        clfs.append(Classifier)
    return(clfs)      
    
def regressors(Name, X_train,y_train):
    regs = []
    for i in range(len(Name)):
        regressor = globals()[Name[i]]
        print(regressor)
        Regressor = regressor()
        Regressor.fit(X_train, y_train)
        regs.append(Regressor)
    return(regs)
    
    
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

acc_conf = []

max_acc = []

for param in params:
    print(param)
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    list_6 = []
    
     
    # Splitting into Test and Train set
    X_train, X_test, y_train, y_test = train_test_split(Xi, y[param], test_size = 0.3, random_state = 0) 
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    
    count = [(list(y_train[param])).count(x) for x in list(set(list(y_train[param])))]

    class_weights = dict(zip(list(set(list(y_train[param]))),count))
    
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
        
    #CLASSIFICATION
    # Fitting Classifier to the Training set
    Classifiers = classifiers(Name_c,X_train,y_train)
    
    Regressors = pd.DataFrame()
    Regressors[0] = regressors(Name_r,X_train_1,y_train_1)
    Regressors[1] = regressors(Name_r,X_train_2,y_train_2)
    Regressors[2] = regressors(Name_r,X_train_3,y_train_3)
    Regressors[3] = regressors(Name_r,X_train_4,y_train_4)
    Regressors[4] = regressors(Name_r,X_train_5,y_train_5)
    Regressors[5] = regressors(Name_r,X_train_6,y_train_6)
    
    Regressors = Regressors.values.tolist()

    
    for clf in  Classifiers :
        try:
            classifier = clf
            classifier.fit(X_train, y_train)
            
            # Predicting the Test set results
            y_pred = classifier.predict(X_test)
            y_predl = list(y_pred)
            y_pred = pd.DataFrame(y_predl)
            
            # Making the Confusion Matrix
            cm = confusion_matrix(list(y_test[param]), y_predl)
            acc = accuracy_score(list(y_test[param]), y_predl)
            # Splitting the train set into specific labels for Regression Training
        
            testlist_1 = []
            testlist_2 = []
            testlist_3 = []
            testlist_4 = []
            testlist_5 = []
            testlist_6 = []
            
        
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
            
            
            for reg in Regressors:
                
                # REGRESSION
                # Training the Regressors
                
                reg_1 = reg[0]
                reg_1.fit(X_train_1, y_train_1)
                
                reg_2 = reg[1]
                reg_2.fit(X_train_2, y_train_2)
                
                reg_3 = reg[2]
                reg_3.fit(X_train_3, y_train_3)
                
                reg_4 = reg[3]
                reg_4.fit(X_train_4, y_train_4)
                
                reg_5 = reg[4]
                reg_5.fit(X_train_5, y_train_5)
                
                reg_6 = reg[5]
                reg_6.fit(X_train_6, y_train_6)
                
                
                y_pred_1 = reg_1.predict(X_test_1)
                y_pred_2 = reg_2.predict(X_test_2)
                y_pred_3 = reg_3.predict(X_test_3)
                y_pred_4 = reg_4.predict(X_test_4)
                y_pred_5 = reg_5.predict(X_test_5)
                y_pred_6 = reg_6.predict(X_test_6)
                
                
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
                c_loop = Classifiers.index(clf)
                r_loop = Regressors.index(reg)
                acc_conf.append([param,c_loop,r_loop,wmape])
                print(str(c_loop) + ' , ' + str(r_loop))
        except:
            continue



np.savetxt("acc_31clf+37reg.csv",  
               acc_conf, 
               delimiter =", ",  
               fmt ='% s') 

np.savetxt("regressors.csv",
           Regressors,
           delimiter =", ",  
           fmt ='% s') 

np.savetxt("classifiers.csv",
           Classifiers,
           delimiter =", ",  
           fmt ='% s') 
           



import pandas as pd
import numpy as np


data = pd.DataFrame(acc_conf)

bw_list = []
gain_list = []
vswr_list = []
bws = []
gains = []
vswrs = []

bw = []
gain = []
vswr = []

for i in range(len(data)):
    if data[0][i] == 'bandwidth':
        bw_list.append(data[3][i])
        bws.append(i)
    elif data[0][i] == 'gain':
        gain_list.append(data[3][i])
        gains.append(i)
    elif data[0][i] == 'vswr':
        vswr_list.append(data[3][i])
        vswrs.append(i)
bw_min = min(bw_list)
gain_min = min(gain_list)
vswr_min = min(vswr_list)
       

bw_ = []
gain_ = []
vswr_ = []

for i in bw_list:
    try:
        bw_.append(i[0])
    except:
        bw_.append(i)
        
for i in gain_list:
    try:
        gain_.append(i[0])
    except:
        gain_.append(i)
        
for i in vswr_list:
    try:
        vswr_.append(i[0])
    except:
        vswr_.append(i)
        
bw_ = avgfit(bw_)
gain_ = avgfit(gain_)
vswr_ = avgfit(vswr_)
        
bw_list = bw_
gain_list = gain_
vswr_list = vswr_

    
sort_bw = sorted(bw_list)
sort_bw_ind = []
for i in sort_bw:
    sort_bw_ind.append(bw_list.index(i))

sort_gain = sorted(gain_list)
sort_gain_ind = []
for i in sort_gain:
    sort_gain_ind.append(gain_list.index(i)  + gains[0] )

sort_vswr = sorted(vswr_list)
sort_vswr_ind = []
for i in sort_vswr:
    sort_vswr_ind.append(vswr_list.index(i)  + vswrs[0])
        

bw = data.iloc[sort_bw_ind[:30]]
gain = data.iloc[sort_gain_ind[:50]]
vswr = data.iloc[sort_vswr_ind[:30]]

bw_clf_list = []
bw_clf_list.append(list(bw[1]))
bw_clf_list = list(((bw_clf_list[0])))
gain_clf_list = []
gain_clf_list.append(list(gain[1]))
gain_clf_list = list(((gain_clf_list[0])))
vswr_clf_list = []
vswr_clf_list.append(list(vswr[1]))
vswr_clf_list = list(((vswr_clf_list[0])))
            

bw_reg_list = []
bw_reg_list.append(list(bw[2]))
bw_reg_list = list(((bw_reg_list[0])))
gain_reg_list = []
gain_reg_list.append(list(gain[2]))
gain_reg_list = list(((gain_reg_list[0])))
vswr_reg_list = []
vswr_reg_list.append(list(vswr[2]))
vswr_reg_list = list(((vswr_reg_list[0])))


#Best Regs
best_bw_reg = []
for i in bw_reg_list:
    if i not in best_bw_reg:
        best_bw_reg.append(i)
best_bw_reg = best_bw_reg[:5]

best_gain_reg = []
for i in gain_reg_list:
    if i not in best_gain_reg:
        best_gain_reg.append(i)
best_gain_reg = best_gain_reg[:5]

best_vswr_reg = []
for i in vswr_reg_list:
    if i not in best_vswr_reg:
        best_vswr_reg.append(i)
best_vswr_reg = best_vswr_reg[:5]

#Best Clfs
best_bw_clf = []
for i in bw_clf_list:
    if i not in best_bw_clf:
        best_bw_clf.append(i)
best_bw_clf = best_bw_clf[:5]

best_gain_clf = []
for i in gain_clf_list:
    if i not in best_gain_clf:
        best_gain_clf.append(i)
best_gain_clf = best_gain_clf[:5]

best_vswr_clf = []
for i in vswr_clf_list:
    if i not in best_vswr_clf:
        best_vswr_clf.append(i)
best_vswr_clf = best_vswr_clf[:5]


#BEST REGRESSORS List
bw_regs = []
for i in best_bw_reg:
    bw_regs.append(Regressors[i][0])

gain_regs = []
for i in best_gain_reg:
    gain_regs.append(Regressors[i][0])
    
vswr_regs = []
for i in best_vswr_reg:
    vswr_regs.append(Regressors[i][0])

#BEST CLASSIFIERS List
bw_clfs = []
for i in best_bw_clf:
    bw_clfs.append(Classifiers[i])

gain_clfs = []
for i in best_gain_clf:
    gain_clfs.append(Classifiers[i])
    
vswr_clfs = []
for i in best_vswr_clf:
    vswr_clfs.append(Classifiers[i])





best_reg = []
for i in reg_list:
    for j in i:
        if j not in best_reg:
            best_reg.append(j)
            
best_regs = []
for i in best_reg:
    best_regs.append(Regressors[i][0])

best_clfs = []
for i in best_clf:
    best_clfs.append(Classifiers[i])




bw = data.iloc[bw_list]

'''
bw

             0   1  2         3
3    bandwidth   0  3  0.402822
40   bandwidth   4  3  0.493311
188  bandwidth  10  3  0.499435
37   bandwidth   4  0  0.541678
0    bandwidth   0  0  0.561167
39   bandwidth   4  2  0.589957
1    bandwidth   0  1  0.591284
185  bandwidth  10  0  0.597837
38   bandwidth   4  1  0.623777
187  bandwidth  10  2  0.627169


gain

        0  1   2        3
667  gain  7   1    2.178
681  gain  7  15  2.26442
669  gain  7   3  2.34512
682  gain  7  16    2.616
694  gain  7  28   2.7604
556  gain  4   1  2.78172
558  gain  4   3   2.8398
697  gain  7  31   2.8422
670  gain  7   4  2.86276
690  gain  7  24  2.86445


vswr

         0   1   2        3
1014  vswr   4  15  1.07083
1002  vswr   4   3  1.11066
1001  vswr   4   2   1.1145
999   vswr   4   0  1.11623
1000  vswr   4   1  1.11849
1532  vswr  26  15  1.30603
1518  vswr  26   1  1.32364
1520  vswr  26   3  1.33734
1003  vswr   4   4  1.34034
1517  vswr  26   0  1.37214



