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



# Getting the exact list 

bw_list = []
gain_list = []
vswr_list = []
bws = []
gains = []
vswrs = []
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

bw_ind = bw_list.index(bw_min)
gain_ind = gain_list.index(gain_min)
vswr_ind = vswr_list.index(vswr_min)

bw = data.iloc[bw_ind]
gain = data.iloc[gain_ind + int(len(data)/3)]
vswr = data.iloc[vswr_ind + int(len(data)*2/3)]

bw = data.iloc[bw_ind]
gain = data.iloc[gain_ind + gains[0]]
vswr = data.iloc[vswr_ind + vswrs[0]]
