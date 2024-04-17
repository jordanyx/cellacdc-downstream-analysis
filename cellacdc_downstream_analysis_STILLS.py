# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:46:31 2023

@author: jyxiao

cellacdc_downstream_analysis_STILLS

Only uses CellACDC built-in background correction. Works best with brightly illuminated stuff.

"""

# In[]: IMPORT STATEMENTS
# keep exposed
    
from os import path #importing just path from os to make it more efficient
import sys
sys.path.append(r"C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis")
# import glob
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 30)
pd.set_option("display.max_colwidth", 150)

import matplotlib.pyplot as plt
import seaborn as sns

from cellacdc import cca_functions
from cellacdc import myutils

import cellacdcAnalysisUtils as acdc_utils

# In[]: MISC USER INPUTS

frame_interval_minutes = 6 # frame interval of timecourse

overall_filepath = 'G:\My Drive\JX_DATA\scaling stuff for mike\\230915_JX_ML205_SCD_5mpf_yellow800ms_backwardsflow/'
ch_name = 'EGFP'

df_force_recalc = True # whether to recalculate the dataframe in case anything upstream of this script changes

# In[]: LOADING DATA, GUI OPTION

# only load ONE EXPERIMENT at a time! otherwise all data will save into the first loaded experiment's folder
data_dirs, positions = cca_functions.configuration_dialog()
#file_names = [path.basename(data_dirs[0])] # you can grab the outermost folder/file in a path using path.basename
file_names = [path.split(p)[-1] for p in data_dirs] # use this line to load multiple experiments, in lieu of the above
image_folders = [[path.join(data_dir, pos_str, 'Images') for pos_str in pos_list] for pos_list, data_dir in zip(positions, data_dirs)]
# determine available channels based on first(!) position.
# Warn user if one or more of the channels are not available for some positions
first_pos_dir = path.join(data_dirs[0], positions[0][0], 'Images')
first_pos_files = myutils.listdir(first_pos_dir)
channels, warn = cca_functions.find_available_channels(first_pos_files, first_pos_dir)

# In[]: GENERATE OVERALL DATAFRAME
# overall_df is essentially acdc_output.csv with additional columns calculated.
# this is already a function, can leave as is

overall_df, is_timelapse_data, is_zstack_data = cca_functions.calculate_downstream_data(
    file_names,
    image_folders,
    positions,
    channels, 
    force_recalculation = df_force_recalc # if True, recalculates overall_df; use this if anything upstream is changed
)
df_force_recalc = False # reset to false in case this block is run again before turning it back

# In: GENERATE OVERALL DF WITH REL
# overall_df_with_rel adds columns to overall_df for each row (aka cell-object) that includes data of its relative.

# if cell cycle annotations were performed in ACDC, extend the dataframe by a join on each cells relative cell
if 'cell_cycle_stage' in overall_df.columns:
    overall_df_with_rel = cca_functions.calculate_relatives_data(overall_df, channels)
# If working with timelapse data build dataframe grouped by phases
group_cols = [
    'Cell_ID', 'generation_num', 'cell_cycle_stage', 'relationship', 'position', 'file', 
    'max_frame_pos', 'selection_subset', 'max_t'
]
# calculate data grouped by phase only in the case, that timelapse data is available
if is_timelapse_data:
    phase_grouped = cca_functions.calculate_per_phase_quantities(overall_df_with_rel, group_cols, channels)
    # append phase-grouped data to overall_df_with_rel
    overall_df_with_rel = overall_df_with_rel.merge(
        phase_grouped,
        how='left',
        on=group_cols
    )
    
# In[]: GENERATE FIELDS USEFUL FOR BOOK-KEEPING
# keep this exposed and mutable; annotations may depend on user/experiment type
# NOTE: you can also add custom fields to the acdc_output.csv via the GUI

# calculate a unique cell id across files by appending file, position, cell id, generation number (more precisely, this is cell cycle number, since we include generation number)
overall_df_with_rel['cell_unique_id'] = overall_df_with_rel.apply(
    lambda x: f'{x["file"]}_{x["position"]}_Cell_{x["Cell_ID"]}_Gen_{int(x["generation_num"])}', axis=1)

# In[]: SAVE STAGE 1 OUTPUT
# keep exposed, also as a reference for various column names

# @todo: add column filter step; trim stuff like quantities from non-quantified channels (Phase, Htb2-mKate, etc.), x/y moments, centroids, etc.

# keep everything that could be even remotely useful
# needed_cols = [
#     'mutant', 'cell_unique_id', 'file', 'position', 'camera', # identifing info
#     'cell_cycle_stage', 'phase_length', 'phase_length_in_minutes', 'cell_cycle_length_minutes', # time/age info
#     'phase_volume_at_beginning', 'phase_volume_at_end', 'phase_combined_volume_at_end', # volume info
#     'phase_'f'{ch_name}_amount_at_beginning', 'phase_'f'{ch_name}_combined_amount_at_beginning', 'phase_'f'{ch_name}_combined_amount_at_end', # fluo info
#     'phase_'f'{ch_name}_concentration_at_beginning', 'phase_'f'{ch_name}_concentration_at_end'
# ]

# temporary; until needed cols is filled in. for now just take everything
df_all = overall_df_with_rel.copy()

# save but only if one experiment is loaded
if len(file_names) == 1:
    expt_filepath = data_dirs[0]
    df_all.to_csv(path_or_buf = path.join(expt_filepath,file_names[0]+'_stage1_output_all.csv'), index=False) # save to experiment folder
    print('Data saved to experiment folder: ' + expt_filepath)
    df_all.to_csv(path_or_buf = path.join(overall_filepath,file_names[0]+'_stage1_output_all.csv'), index=False) # save to overall folder for all experiments of the same category
    print('Data also saved to overall folder: ' + overall_filepath)

# @todo: uncomment this after figuring out needed columns
#df_gen1 = overall_df_with_rel.loc[gen1_filter, needed_cols].copy()
#df_all.to_csv(path_or_buf = expt_filepath +'/'+file_names[0]+'_stage1_output_gen1.csv', index=False)

# In[]: VERSUS PLOTS

# USER INPUT: filter for cells of interest
filter_idx = (df_all['cell_cycle_stage']=='G1')         \
            &(df_all['is_cell_excluded']==0)         \
            &(df_all['cell_vol_fl']>5)         \
            &(df_all['file']==('230922_JX_ML205_stills488') )        
                
df_filt = df_all.loc[filter_idx].copy()

df_filt['vol_fl_normed'] = df_filt['cell_vol_fl'] / df_filt['cell_vol_fl'].mean()
df_filt['conc_normed'] = df_filt[f'{ch_name}_concentration_autoBkgr_from_vol_fl'] / df_filt[f'{ch_name}_concentration_autoBkgr_from_vol_fl'].mean()

hue = 'file'
x_var = 'vol_fl_normed'
# y_var = f'{ch_name}_amount_autoBkgr'
y_var = 'conc_normed'


sns.set_theme(style='white', font_scale = 3.5)
FIGvs = sns.lmplot(data = df_filt, 
                   x = x_var, y = y_var, 
                   hue = hue, 
                   height=12, aspect=1.2, 
                   # x_bins = 6,
                   line_kws={"lw":3}, scatter_kws={"s": 100}, legend='full', facet_kws={'legend_out':False})
    

# place legend
leg = plt.legend(loc='center',bbox_to_anchor=(1.4,0.5), frameon=False)
# set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(6)

# FIGvs.set(xlim=(0, 200), ylim=(0, 1.1)) # size vs growth

