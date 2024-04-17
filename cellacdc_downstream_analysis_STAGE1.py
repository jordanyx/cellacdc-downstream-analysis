# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:04:53 2022

@author: jyxiao

ACDC OUTPUT ANALYSIS
STAGE 1: PROCESSING (SINGLE EXPERIMENT)

Input: acdc_output.csv for a SINGLE experiment
Output: overall_df_with_rel.csv, which contains a dataframe with all possibly relevant info, saved into same folder as the loaded experiment

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

frame_interval_minutes = 3 # frame interval of timecourse

# fluo info and other user inputs

# overall_filepath = 'G:\My Drive\JX_DATA\Whi5-mCitrine expression\Whi5-mCitrine expression all/' # path for experiment category
# ch_name = 'mCitrineScaledFFC' # name of fluo channel of interest for autofluorescence corrections; must be same as whatever is in the experiment metadata

# overall_filepath = 'G:\My Drive\JX_DATA\CLN2pr expression\CLN2pr expression all/' # path for experiment category
# ch_name = 'VenusScaledFFC' # name of fluo channel of interest for autofluorescence corrections; must be same as whatever is in the experiment metadata

# overall_filepath = 'G:\My Drive\JX_DATA\scaling stuff for mike/'
# ch_name = 'mCitrineScaledFFC'

overall_filepath = 'E:\DATA\JK FKH/JK FKH compiled'
ch_name = 'mCitrineScaledFFC'

# UNGATED PARAMS, 3/9/23
a = 9.947188613556736e-06 # units of au/pix/vox
b = 1.267706924139061 # units of au/pix

# GATED PARAMS, 3/13/23
piecewise_bool = True
# piecewise params (big: >25000)
a_small = 3.365835893138414e-05
b_small = 0.8761850772728765
a_big = 3.991222054267158e-06
b_big = 1.6704754565094582

df_force_recalc = True # whether to recalculate the dataframe in case anything upstream of this script changes

# =============================================================================
# # mCitrine params updated 7/8/22, provided by JK
# a = 5.2615323399390606e-06 # units of au/pix/vox
# b = 4.544630239072624 # units of au/pix
# 
# =============================================================================

# =============================================================================
# # TESTING
# ch_name = 'mCitrineRaw'
# a = 8.868221835189484e-06
# b = 4.544630239072624 + 0.0 # TESTING
# =============================================================================

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

# In[]: LOADING DATA, MANUAL OPTION

# only load ONE EXPERIMENT at a time! otherwise all data will save into the first loaded experiment's folder

# =============================================================================
# data_dirs =  [
#                 'G:\My Drive\JX_DATA/Cln2pr expression/220502_SM_JX61a', # WT clone 1 expt 1
#               ]
# 
# positions = [
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
#             ]
# 
# =============================================================================
# =============================================================================
# data_dirs = [
#                 'E:\DATA\Cln2pr expression/220502_SM_JX61a', # WT clone 1 expt 1
#                 'E:\DATA\Cln2pr expression/220610_JX_JX61a', # WT clone 1 expt 2
#                 'E:\DATA\Cln2pr expression/220608_JX_JX61b', # WT clone 2 expt 1
#                 'E:\DATA\Cln2pr expression/220616_JX_JX61b', # WT clone 2 expt 2
#     
#                 'E:\DATA\Cln2pr expression/220609_JX_JX62b', # 12A clone 1 expt 1
#                 'E:\DATA\Cln2pr expression/220617_JX_JX62b', # 12A clone 1 expt 2
#                 'E:\DATA\Cln2pr expression/220515_JX_JX62c', # 12A clone 2 expt 1
#                 'E:\DATA\Cln2pr expression/220610_JX_JX62c', # 12A clone 2 expt 2
#     
#                 'E:\DATA\Cln2pr expression/220522_JX_JX63a', # 7A clone 1 expt 1
#                 'E:\DATA\Cln2pr expression/220611_JX_JX63a', # 7A clone 1 expt 2
#                 'E:\DATA\Cln2pr expression/220516_JX_JX63b', # 7A clone 2 expt 1
#                 'E:\DATA\Cln2pr expression/220613_JX_JX63b', # 7A clone 2 expt 2
#                 
#                 'E:\DATA\Cln2pr expression/220712_JX_JX68a', # 19Av2 clone 1 expt 1
#                 'G:\My Drive\JX_DATA/Cln2pr expression/220713_JX_JX68b', # 19Av2 clone 2 expt 1
#     
#                 'G:\My Drive\JX_DATA/Cln2pr expression/220505_JX_JX64a', # 5S clone 1 expt 1
#                 'G:\My Drive\JX_DATA/Cln2pr expression/220523_JX_JX69a', # 7T clone 1 expt 1
#                 'G:\My Drive\JX_DATA/Cln2pr expression/220522_JX_JX50b', # Whi5 del
#             ]
# 
# positions = [
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5'],
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5'],
#     
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7','Position_8'],
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6'],
#     
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5'],
#                 
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5'],
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5'],
#     
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6'],
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6'],
#             ]
# 
# file_names = [path.split(p)[-1] for p in data_dirs]
# image_folders = [[path.join(data_dir, pos_str, 'Images') for pos_str in pos_list] for pos_list, data_dir in zip(positions, data_dirs)]
# # determine available channels based on first(!) position.
# # Warn user if one or more of the channels are not available for some positions
# first_pos_dir = path.join(data_dirs[0], positions[0][0], 'Images')
# first_pos_files = myutils.listdir(first_pos_dir)
# channels, warn = cca_functions.find_available_channels(first_pos_files, first_pos_dir)
# 
# =============================================================================

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

if ch_name.__contains__('mCitrine'):
    # determine which mutant by checking what substring filename contains
    overall_df_with_rel['mutant'] = overall_df_with_rel.apply(
        lambda x: 'WT-3mpf' if x.loc['file'].__contains__('MS358_3mpf') \
            else ('7A-3mpf' if x.loc['file'].__contains__('JX27a_3mpf') \
        # lambda x: 'WT' if x.loc['file'].__contains__('MS358_3mpf') \
            # else ('7A' if x.loc['file'].__contains__('JX27a_3mpf') \
            else ('WT' if x.loc['file'].__contains__('MS358') \
            else ('12A' if x.loc['file'].__contains__('JX26') \
            else ('12A' if x.loc['file'].__contains__('JX101') \
            else ( '7A' if x.loc['file'].__contains__('JX27') \
            else ( '5S' if x.loc['file'].__contains__('JX28') \
            else ('19A' if x.loc['file'].__contains__('JX66') \
            else ( '7T' if x.loc['file'].__contains__('JX60') \
            else ('12A x2' if x.loc['file'].__contains__('JX72') \
            else ( '7A x2' if x.loc['file'].__contains__('JX83') \
            else ( '6A' if x.loc['file'].__contains__('JX126') \
            else (  'WT-NLS' if x.loc['file'].__contains__('JX127') \
            else ( '12A-NLS' if x.loc['file'].__contains__('JX128') \
            else 'mutant error' )))))))))))))
        ,axis=1)

elif ch_name.__contains__('Venus'):
    # determine which mutant by checking what substring filename contains
    overall_df_with_rel['mutant'] = overall_df_with_rel.apply(
        lambda x:  'WT' if x.loc['file'].__contains__('JX61') \
            else ('12A' if x.loc['file'].__contains__('JX62') \
            else ( '7A' if x.loc['file'].__contains__('JX63') \
            else ( '5S' if x.loc['file'].__contains__('JX64') \
            else ('19A' if x.loc['file'].__contains__('JX68') \
            else ( '7T' if x.loc['file'].__contains__('JX69') \
            else ('12A x2' if x.loc['file'].__contains__('JX99') \
            else ( 'Whi5 del' if x.loc['file'].__contains__('JX50') \
            else 'mutant error' )))))))
        ,axis=1)
else:
    print('mutant labels unspecified')

# # identify which camera by figuring out the date cutoff (April 15 2022)
# overall_df_with_rel['camera'] = overall_df_with_rel.apply(
#     lambda x:  'old' if int(x.loc['file'][0:6]) < 220415 
#         else   'new' 
#     ,axis=1)

# In[]: CALCULATE VARIOUS USEFUL QUANTITIES
# NOTE: some of these may become redundant if some calculations are moved to the ACDC GUI save step instead
# in particular, autofluo will probably be moved upstream

# phase timing quantities (phase lengths, time in phase)
overall_df_with_rel = acdc_utils.calculate_phase_timing_quantities(overall_df_with_rel, frame_interval_minutes)

# autofluorescence corrections (fluo amount, fluo concentration)
overall_df_with_rel = acdc_utils.calculate_autofluo_corrected_quantities(overall_df_with_rel, ch_name, a, b, piecewise_bool, a_small, b_small, a_big, b_big)

# combined mother/bud quantities (volume, fluo amount, fluo concentration)
overall_df_with_rel = acdc_utils.calculate_combined_mother_bud_quantities(overall_df_with_rel, ch_name)

# quantities relevant for complete cell cycles (birth/budding/division stats, phase/cycle lengths)
overall_df_with_rel = acdc_utils.calculate_complete_cycle_quantities(overall_df_with_rel, frame_interval_minutes)


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


'''
END OF STAGE 1 ANALYSIS; MOVE TO STAGE 2 FROM HERE.
'''

# In[]: SANITY CHECK PLOTTING (OPTIONAL TO RUN)
# use to make sure single cell traces for a given position look right, or that positions within experiment are similar
# keep this exposed and simple to use

# USER INPUT: filter for cells of interest
filter_idx = (df_all['is_cell_excluded']==0)         \
            &(df_all['is_cell_dead']==0)         \
            &(df_all['complete_cycle']==1)         \
            &(df_all['generation_num']==1)         \
                &(df_all['position'] == 'Position_5')  \
                
df_filt = df_all.loc[filter_idx].copy()

# choose fields to plot
x_var = 'bud_aligned_time_in_minutes'
# y_var = 'mCitrineFFC_CV'
# y_var = 'Combined m&b volume fL'
y_var = 'Combined m&b amount'
# y_var = 'Combined m&b concentration'
xlim = [-120,120]
ylim = [0,None]

# overlaying by mutant, position, etc.
# hue = 'file'
hue = 'position'
hue = 'cell_unique_id'

sns.set_theme(style='whitegrid', font_scale = 3)

# the line that does the actual plotting
FIG = sns.relplot(data=df_filt, x=x_var, y=y_var, kind='line', hue=hue, height=12, aspect=1.2, legend='full', facet_kws={'legend_out': True})

# setting axis limits
FIG.set(xlim=xlim)
FIG.set(ylim=ylim)

# print number of cells (technically cell cycles)
types = list(set(df_filt[hue]))
num_types = len(types)
totcells = 0
for i in range(0,num_types):
    data_temp = df_filt.loc[df_filt[hue]==types[i]].copy()
    numcells = len(set(data_temp['cell_unique_id']))
    print(types[i] + ', n = ' + str(numcells))
    totcells = totcells + numcells

print('Total: N = ' + str(totcells))

# title and axis labels
#FIG.fig.suptitle('Whi5-mCitrine localization')

FIG.axes[0,0].set_xlabel('Time aligned to budding (min)')
#FIG.axes[0,0].set_ylabel(y_var)
# FIG.axes[0,0].set_ylabel('Coeff. of variation')
FIG.axes[0,0].set_ylabel('Whi5-mCitrine amount (a.u.)')
# FIG.axes[0,0].set_ylabel('mVenus-PEST conc. (a.u.)')

if (len(file_names) == 1) & (hue == 'position'):
    FIG.savefig(path.join(expt_filepath,file_names[0]+'_by_position.png'))
    print('Figure saved to expt folder.')


# =============================================================================
# # In[]: VERSUS PLOTS
# 
# # USER INPUT: filter for cells of interest
# filter_idx = (df_all['frame_i']==120)         \
#             &(df_all['cell_cycle_stage']=='G1')         \
#             &(df_all['is_cell_excluded']==0)         \
#             &(df_all['is_cell_dead']==0)         \
#                 # &(df_all['position']=='Position_6')         \
#                 
# df_filt = df_all.loc[filter_idx].copy()
# 
# hue = 'file'
# x_var = 'Combined m&b volume fL'
# y_var = 'Combined m&b concentration'
# 
# # x_var = 'G1 length minutes'
# # y_var = 'G1 relative growth'
# # df['G1 relative growth'] = np.log(df['Combined m&b volume budding'] / df['Combined m&b volume birth'])
# 
# # df_filt_nodups['SG2M relative growth'] = np.log(df_filt_nodups['Combined m&b volume division'] / df_filt_nodups['Combined m&b volume budding'])
# # x_var = 'SG2M length minutes'
# # y_var = 'SG2M relative growth'
# 
# # x_var = 'Combined m&b volume birth'
# # y_var = 'Combined m&b volume budding'
# 
# # TO DO: add binning here
# 
# sns.set_theme(style='white', font_scale = 3.5)
# FIGvs = sns.lmplot(data = df_filt, 
#                    x = x_var, y = y_var, 
#                    hue = hue, 
#                    height=12, aspect=1.2, 
#                    # x_bins = 6,
#                    line_kws={"lw":3}, scatter_kws={"s": 100}, legend='full', facet_kws={'legend_out':False})
#     
# 
# # place legend
# leg = plt.legend(loc='center',bbox_to_anchor=(1.5,0.5), frameon=False)
# # set the linewidth of each legend object
# for legobj in leg.legendHandles:
#     legobj.set_linewidth(6)
# 
# # FIGvs.set(xlim=(0, None), ylim=(0, 2000))
# 
# =============================================================================

# In[]: TESTING CALCULATONS


