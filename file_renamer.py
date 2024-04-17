# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:14:38 2023

RENAME MISNAMED FILES

@author: jyxiao
"""

import os
import sys
sys.path.append(r"C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis")

from pathlib import Path

# In[]: MISC USER INPUTS

# NOTE TO SELF: be careful! not easy to undo any changes made

overall_filepath = r"E:\DATA\JK FKH\240324_JX_JK137_0aTc"

target_to_replace = 'temp'
replacement = 'mCitrineRaw'

file_iterator = Path(overall_filepath).rglob('*' + target_to_replace + '*')

for file in file_iterator:
    print('Fixing: ' + file.name)
    
    newname =  str(file).replace(target_to_replace, replacement)
    if newname != file:
        os.rename(file,newname)
        
    print('Fixed to: ' + newname)
        
        