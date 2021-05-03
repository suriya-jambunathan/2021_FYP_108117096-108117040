# Python code to predict the Bandwidth, Gain and VSWR of an antenna, given the dimensions of the antenna, using the trained Class_Reg Model
import pandas as pd
import numpy as np
from itertools import compress
import sys
import joblib

sys.path.append('MLANT/')

from classes import *

df = pd.read_csv('Content/antenna.csv')

feat_inds = [[True, False, True, True, True, True, True],
             [True, False, True, False, True, True, True],
             [True, False, True, True, True, True, True]]

best = [2141.3747, 650.78, 214.13747, 77.0894892, 7, 1200, 2141.3747]

def predict(data):
    prediction = []
    for i in range(3):
        prediction.append((model[i]).predict(list(compress(data, feat_inds[i] ))))
    return(prediction)
  
 
#user_input = [ .... ] Enter the dimensions of the antenna here


model = joblib.load('Saved/trained_class_reg.sav')
model = model[:3]

#Store model predictions in a variable
prediction = predict(user_input)



pred_df = pd.DataFrame()

pred_df['Parameter'] = ['Bandwidth', 'Gain', 'VSWR']
pred_df['Prediction'] = [prediction[0], prediction[1], prediction[2]]

print(pred_df)
    
