# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 13:46:51 2022

@author: jyxiao

ACDC OUTPUT ANALYSIS
STAGE 2: COMPILING MULTIPLE EXPERIMENTS

Input: STAGE 1 output for MANY experiments
Output: two saved dataframes, one for time-dependent quantities and one for per-phase quantities (including newly calculated ones potentially), 
which compiles data from multiple experiments

NOTE: It would not be unreasonable to keep a version of this script for every major experiment category.
For example, Jordan wants to compile all the Cln2pr expression data into one mega-file, and have a separate one for Whi5-mCit data.
To add to the mega-file, it's probably easiest to keep the manual data load option and add a new directory to add to the compiled one.
Maybe we should write something to append without having to reload everything?

"""
# In[]: IMPORT STATEMENTS
# keep exposed

import os
import sys
sys.path.append(r"C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis")
# import glob
# import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 30)
pd.set_option("display.max_colwidth", 150)

import seaborn as sns
sns.set_theme()
from cellacdc import cca_functions
from cellacdc import myutils

#import cellacdcAnalysisUtils as acdc_utils

# In[]: MISC USER INPUTS
# files to load, fluo channel name
ch_name = 'Venus'

# In[]: LOADING DATA, GUI OPTION

data_dirs, positions = cca_functions.configuration_dialog()
file_names = [os.path.split(p)[-1] for p in data_dirs]
image_folders = [[os.path.join(data_dir, pos_str, 'Images') for pos_str in pos_list] for pos_list, data_dir in zip(positions, data_dirs)]
# determine available channels based on first(!) position.
# Warn user if one or more of the channels are not available for some positions
first_pos_dir = os.path.join(data_dirs[0], positions[0][0], 'Images')
first_pos_files = myutils.listdir(first_pos_dir)
channels, warn = cca_functions.find_available_channels(first_pos_files, first_pos_dir)

# In[]: LOADING DATA, MANUAL OPTION

data_dirs = [
                'C:/Users/jyxiao/DATA/Cln2pr expression/220502_SM_JX61a', # WT clone 1 expt 1
                'C:/Users/jyxiao/DATA/Cln2pr expression/220610_JX_JX61a', # WT clone 1 expt 2
                'C:/Users/jyxiao/DATA/Cln2pr expression/220608_JX_JX61b', # WT clone 2 expt 1
                'C:/Users/jyxiao/DATA/Cln2pr expression/220616_JX_JX61b', # WT clone 2 expt 2
    
                'C:/Users/jyxiao/DATA/Cln2pr expression/220609_JX_JX62b', # 12A clone 1 expt 1
                'C:/Users/jyxiao/DATA/Cln2pr expression/220617_JX_JX62b', # 12A clone 1 expt 2
                'C:/Users/jyxiao/DATA/Cln2pr expression/220515_JX_JX62c', # 12A clone 2 expt 1
                'C:/Users/jyxiao/DATA/Cln2pr expression/220610_JX_JX62c', # 12A clone 2 expt 2
    
                'C:/Users/jyxiao/DATA/Cln2pr expression/220522_JX_JX63a', # 7A clone 1 expt 1
                'C:/Users/jyxiao/DATA/Cln2pr expression/220611_JX_JX63a', # 7A clone 1 expt 2
                'C:/Users/jyxiao/DATA/Cln2pr expression/220516_JX_JX63b', # 7A clone 2 expt 1
                'C:/Users/jyxiao/DATA/Cln2pr expression/220613_JX_JX63b', # 7A clone 2 expt 2
                
                'C:/Users/jyxiao/DATA/Cln2pr expression/220712_JX_JX68a', # 19Av2 clone 1 expt 1
    
                'C:/Users/jyxiao/DATA/Cln2pr expression/220505_JX_JX64a', # 5S clone 1 expt 1
                'C:/Users/jyxiao/DATA/Cln2pr expression/220523_JX_JX69a', # 7T clone 1 expt 1
                'C:/Users/jyxiao/DATA/Cln2pr expression/220522_JX_JX50b', # Whi5 del
            ]

positions = [
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5'],
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5'],
    
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7','Position_8'],
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6'],
    
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5'],
                
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5'],
    
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6'],
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6'],
            ]


file_names = [os.path.split(path)[-1] for path in data_dirs]
image_folders = [[os.path.join(data_dir, pos_str, 'Images') for pos_str in pos_list] for pos_list, data_dir in zip(positions, data_dirs)]
# determine available channels based on first(!) position.
# Warn user if one or more of the channels are not available for some positions
first_pos_dir = os.path.join(data_dirs[0], positions[0][0], 'Images')
first_pos_files = myutils.listdir(first_pos_dir)
channels, warn = cca_functions.find_available_channels(first_pos_files, first_pos_dir)


# In[]: LOAD STAGE 1 OUTPUT

# loop over data_dirs and load the stage 1 output for each one
# then append the csvs

df_compiled_all = pd.concat((pd.read_csv(f) for f in all_files))


# some function that loads multiple dataframes, one per experiment to be compiled

# In[]: COMPILE EXPERIMENTS AND FILTER COLUMNS/ROWS
# combine multiple experiments into one dataframe that takes all the data worth plotting (to make things simpler to plot later)
# then make a copy of said dataframe with time-dependent data removed and duplicates removed, to create a dataframe with only per-phase quantities

# synthesis rate calculations go here? though that could probably go in stage 1


# select needed cols from overall_df_with_rel; for this part, make sure to only take fields that are constant for a given cell cycle
needed_cols = [
    'mutant', 'cell_unique_id', 'file', 'position', 'camera', # identifing info
    'cell_cycle_stage', 'phase_length', 'phase_length_in_minutes', 'cell_cycle_length_minutes', # time/age info
    'phase_volume_at_beginning', 'phase_volume_at_end', 'phase_combined_volume_at_end', # volume info
    'phase_'f'{ch_name}_amount_at_beginning', 'phase_'f'{ch_name}_combined_amount_at_beginning', 'phase_'f'{ch_name}_combined_amount_at_end', # fluo info
    'phase_'f'{ch_name}_concentration_at_beginning', 'phase_'f'{ch_name}_concentration_at_end'
]

# dataframe containing only per-phase quantities (so each cell cycle is represented only once in this dataframe)
df_perphase = overall_df_with_rel.loc[filter_idx, needed_cols].copy().drop_duplicates()

# In[]: SAVE STAGE 2 OUTPUT

# two save lines, one for time-dep and one for per-phase info
# can also add a few other saves for other forms of filtering, like first-gen mothers only

# END OF STAGE 2; MOVE TO STAGE 3 FOR PLOTTING