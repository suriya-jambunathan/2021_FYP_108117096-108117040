# Python code to train the genetic algorithm in conjunction with class_reg algorithm

import numpy as np
import pandas as pd
import random
import os

import sys
sys.path.append('MLANT/')
from GeneticAlgorithm import *
from classes import * 

data = pd.read_csv('./Dataset/antenna.csv')

X_list = list(data.columns)[:7]
params= ['bandwidth', 'gain', 'vswr']

for param in params:
    
    X_train, X_test, y_train, y_test = train_test_split(data[X_list],
                                                        pd.DataFrame(data[param]),
                                                        test_size = 0.3,
                                                        random_state = None)
    
    model = class_reg(data, X_list, param, 0.2,0.2, 1)
    
    y_test = processing.avgfit(processing.tolist(y_test))
    
    obj = GeneticAlgorithm(X_test=X_test,y_train=y_train,y_test=y_test,
                           size=100,n_feat=7,n_parents=100,mutation_rate=0.001,
                           n_gen=25,X_train=X_train,model = model)
    
    chromo, score = obj.fit()
    
    ind = score.index(min(score))
        
    model.fit(X_train.iloc[:,chromo[ind]],y_train_og)
    predictions = model.predict(X_totest.iloc[:,chromo[ind]])
    
    mse = mean_squared_error(list(y_totest[0]),list(predictions))
    mse = (np.sqrt(mse) - min(data[param]))/(max(data[param]) - min(data[param]))
    mse = mse**2
    
    wmape = metric.wmape(list(y_totest[0]),list(predictions))
    
    print(param)
    print("\Mean Squared Error  after genetic algorithm is = "+str(mse))
    print("\Weighted Mean Absolute Percentage Error after genetic algorithm is = "+str(wmape))

