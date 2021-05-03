#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 08:54:31 2021

@author: suriyaprakashjambunathan
"""

import os 
filename_1 = 'Regressors.py'

filename = filename_1
exec(compile(open(filename, "rb").read(), filename, 'exec'))

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

filename_2 = 'Classifiers.py'

filename = filename_2
exec(compile(open(filename, "rb").read(), filename, 'exec'))

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

filename_3 = 'use_saved_class+reg.py'

filename = filename_3
exec(compile(open(filename, "rb").read(), filename, 'exec'))

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


