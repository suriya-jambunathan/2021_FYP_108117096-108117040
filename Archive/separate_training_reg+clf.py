#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 07:59:39 2021

@author: suriyaprakashjambunathan
"""

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
    l = len(y_true)
    num = 0
    den = 0
    for i in range(l):
        num = num + (abs(y_pred[i] - y_true[i]))
        den = den + y_true[i]
    return abs(num/den) * 100

def regressors(Name, X_train,y_train):
    regs = []
    for i in range(len(Name)):
        regressor = globals()[Name[i]]
        print(regressor)
        Regressor = regressor()
        Regressor.fit(X_train, y_train)
        regs.append(Regressor)
    return(regs)
    
def classifiers(Name, X_train,y_train):
    clfs = []
    for i in range(len(Name)):
        classifier = globals()[Name[i]]
        print(classifier)
        Classifier = classifier()
        Classifier.fit(X_train, y_train)
        clfs.append(Classifier)
    return(clfs)      
    
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
            
gain =avgfit(list(dataset['gain']))
dataset['gain'] = gain

vswr =avgfit(list(dataset['vswr']))
dataset['vswr'] = vswr

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


acc_conf = []

max_acc = []

for param in params:
    print(param)
    # Splitting into Test and Train set
    X_train, X_test, y_train, y_test = train_test_split(Xi, y[param], test_size = 0.3, random_state = 0) 
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    #print(name_r)
    Regressor = regressors(Name_r,X_train,y_train)
    for reg in Regressor :
        y_pred = reg.predict(X_test)
        wmape = mean_absolute_percentage_error(list(y_test[param]), list(y_pred))
        if not np.isnan(wmape):
            try:
                acc_conf.append([param, reg, wmape[0]])
            except:
                acc_conf.append([param, reg, wmape])
               

wmape = pd.DataFrame(acc_conf)

wmape.to_csv('regressors_wmape.csv')






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
     
    # Splitting into Test and Train set
    X_train, X_test, y_train, y_test = train_test_split(Xi, y[param], test_size = 0.3, random_state = 0) 
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    
    count = [(list(y_train[param])).count(x) for x in list(set(list(y_train[param])))]

    class_weights = dict(zip(list(set(list(y_train[param]))),count))
            
    #CLASSIFICATION
    # Fitting Classifier to the Training set
    Classifiers = classifiers(Name_c,X_train,y_train)
    for clf in Classifiers:
        try:
            y_pred = clf.predict(X_test)
            y_predl = list(y_pred)
            y_pred = pd.DataFrame(y_predl)
            # Making the Confusion Matrix
            acc = accuracy_score(list(y_test[param]), y_predl)
            
            acc_conf.append([param, clf, acc*100])
        except:
            continue
        
acc = pd.DataFrame(acc_conf)

acc.to_csv('classifiers_acc.csv')