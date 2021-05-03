#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:29:12 2021

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


def plot(df):
    xs = list(df[1])
    ys = list(df[2])
    zs = list(df[3])
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(xs, ys, zs)
    plt.show()
     
    
def nospace(string): 
  
    # To keep track of non-space character count 
    count = 0
  
    list = [] 
  
    # Traverse the given string. If current character 
    # is not space, then place it at index 'count++' 
    for i in range(len(string)): 
        if string[i] != ' ' and string[i] != '[' and string[i] != ']': 
            list.append(string[i]) 
  
    return toString(list) 
  
def toString(List): 
    return ''.join(List) 


def search(param,i,j):
    ans = data.loc[data[0] == param]
    ans = ans.loc[ans[1] == i]
    ans = ans.loc[ans[2] == j]
    ret = pd.DataFrame(ans[3])
    ret[0] = ret[3]
    del ret[3] 
    ret = ret[0].iloc[0]
    return(ret)


def create_df(param):
    if param == 'bandwidth':
        clf_list = list(set(bw[1]))
        reg_list = list(set(bw[2]))
    elif param == 'gain':
        clf_list = list(set(gain[1]))
        reg_list = list(set(gain[2]))
    elif param == 'vswr':
        clf_list = list(set(vswr[1]))
        reg_list = list(set(vswr[2]))
        
    cols = []
    cols.append(' ')
    for i in reg_list:
        cols.append(Regressors[0][i])
    cols[35] = 'RidgeCV()'
    
    rows = []
    for i in clf_list:
        rows.append(Classifiers[0][i])
    
    df_list = []
    ind = 0
    for i in clf_list:
        cl = []
        cl.append(rows[ind])
        ind = ind + 1
        for j in reg_list:
            cl.append(search(param,i,j))
        df_list.append(cl)
    
    return(pd.DataFrame(df_list, columns = cols))
    

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('acc_31clf+37reg.csv',header = None)
       
        
bw = data.loc[data[0]== 'bandwidth']
gain = data.loc[data[0]== 'gain']
vswr = data.loc[data[0]== 'vswr']


data = pd.read_csv('acc_31clf+37reg.csv',header = None)

data[3] = ([((nospace(i))) for i in data[3]] )

data[3] = [float(i) for i in data[3]]

data[3] = avgfit(data[3])


        
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
        

bw = data.iloc[sort_bw_ind[:50]]
gain = data.iloc[sort_gain_ind[:50]]
vswr = data.iloc[sort_vswr_ind[:50]]


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
        




Regressors = regressors
Classifiers = classifiers

bw = data.iloc[sort_bw_ind]
gain = data.iloc[sort_gain_ind]
vswr = data.iloc[sort_vswr_ind]

###################################

bw_mape = create_df('bandwidth')
gain_mape = create_df('gain')
vswr_mape = create_df('vswr')


bw_mape.to_csv('bandwidth.csv')
gain_mape.to_csv('gain.csv')
vswr_mape.to_csv('vswr.csv')

###################################


bw = data.iloc[sort_bw_ind[:50]]
gain = data.iloc[sort_gain_ind[:50]]
vswr = data.iloc[sort_vswr_ind[:50]]



    
x = [0,4,10,7,26]
y = [0,1,2,3,16]

y = [28]
for i in x:
    for j in y:
        print(str(i) + ', ' + str(j) + '   ' + str(search('bandwidth',i,j)))
        print('   ')
      
        
x = [0,4,7,11,6]
y = [0,1,3,15,16]
y = [28]

for i in x:
    for j in y:
        print(str(i) + ', ' + str(j) + '   ' + str(search('gain',i,j)))
        print('   ')
        

x = [0,4,7,26,10]
y = [2,0,15,1,3]
y = [8]

for i in x:
    for j in y:
        print(str(i) + ', ' + str(j) + '   ' + str(search('vswr',i,j)))
        print('   ')




regressors = pd.read_csv('regressors.csv',header = None)

regressors = regressors.iloc[0:37]

classifiers = pd.read_csv('classifiers.csv',header = None)

classifiers = pd.DataFrame(classifiers[0])

for i in x:
    print(str(classifiers.iloc[i]))

print('     ')

for j in y:
    print(str((regressors.iloc[j])[0]))





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





from pykeyboard import PyKeyboard




from pykeyboard import InlineKeyboard
from pyrogram.types import InlineKeyboardButton


keyboard = InlineKeyboard(row_width=3)
keyboard.add(
    InlineKeyboardButton('1', 'inline_keyboard#1'),
    InlineKeyboardButton('2', 'inline_keyboard#2'),
    InlineKeyboardButton('3', 'inline_keyboard#3'),
    InlineKeyboardButton('4', 'inline_keyboard#4'),
    InlineKeyboardButton('5', 'inline_keyboard#5'),
    InlineKeyboardButton('6', 'inline_keyboard#6'),
    InlineKeyboardButton('7', 'inline_keyboard#7'))
