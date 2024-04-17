#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
#sys.argv = ['']
sys.path.append('../cellacdc/')
import glob
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 30)
pd.set_option('display.max_colwidth', 150)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
sns.set_theme()
from cellacdc import cca_functions
from cellacdc import myutils

from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:





# # Configurations
# - follow the file selection dialog:
#     - select microscopy folder in first step
#     - select positions of the selected folder in second step
# - repeat to add more positions to the analysis
# - positions selected within one iteration of the dialog will be pooled together in the following analyses

# In[ ]:


data_dirs, positions = cca_functions.configuration_dialog()
file_names = [os.path.split(path)[-1] for path in data_dirs]
image_folders = [[os.path.join(data_dir, pos_str, 'Images') for pos_str in pos_list] for pos_list, data_dir in zip(positions, data_dirs)]
# determine available channels based on first(!) position.
# Warn user if one or more of the channels are not available for some positions
first_pos_dir = os.path.join(data_dirs[0], positions[0][0], 'Images')
first_pos_files = myutils.listdir(first_pos_dir)
channels, warn = cca_functions.find_available_channels(first_pos_files, first_pos_dir)

data_dirs, positions, file_names


# In[4]:


# MANUAL ENTRY OPTION: ALL NEW CAMERA DATA, as of 7/1/22

data_dirs =  [
                'C:/Users/jyxiao/DATA/220502_SM_JX61a', # WT clone 1 expt 1
                'C:/Users/jyxiao/DATA/220610_JX_JX61a', # WT clone 1 expt 2
                'C:/Users/jyxiao/DATA/220608_JX_JX61b', # WT clone 2 expt 1
                'C:/Users/jyxiao/DATA/220616_JX_JX61b', # WT clone 2 expt 2
    
                'C:/Users/jyxiao/DATA/220609_JX_JX62b', # 12A clone 1 expt 1
                'C:/Users/jyxiao/DATA/220617_JX_JX62b', # 12A clone 1 expt 2
                'C:/Users/jyxiao/DATA/220515_JX_JX62c', # 12A clone 2 expt 1
                'C:/Users/jyxiao/DATA/220610_JX_JX62c', # 12A clone 2 expt 2
    
                'C:/Users/jyxiao/DATA/220522_JX_JX63a', # 7A clone 1 expt 1
                'C:/Users/jyxiao/DATA/220611_JX_JX63a', # 7A clone 1 expt 2
                'C:/Users/jyxiao/DATA/220516_JX_JX63b', # 7A clone 2 expt 1
                'C:/Users/jyxiao/DATA/220613_JX_JX63b', # 7A clone 2 expt 2
    
                'C:/Users/jyxiao/DATA/220505_JX_JX64a', # 5S clone 1 expt 1
                'C:/Users/jyxiao/DATA/220523_JX_JX69a', # 19Av2 clone 1 expt 1
                'C:/Users/jyxiao/DATA/220522_JX_JX50b', # Whi5 del
             ]

file_names = [
                '220502_SM_JX61a',
                '220610_JX_JX61a',
                '220608_JX_JX61b',
                '220616_JX_JX61b',
    
                '220609_JX_JX62b',
                '220617_JX_JX62b',
                '220515_JX_JX62c',
                '220610_JX_JX62c',
    
                '220522_JX_JX63a',
                '220611_JX_JX63a',
                '220516_JX_JX63b',
                '220613_JX_JX63b',
    
                '220505_JX_JX64a',
                '220523_JX_JX69a',
                '220522_JX_JX50b',
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
    
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6'],
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6'],
            ]

image_folders = [[os.path.join(data_dir, pos_str, 'Images') for pos_str in pos_list] for pos_list, data_dir in zip(positions, data_dirs)]
# determine available channels based on first(!) position.
# Warn user if one or more of the channels are not available for some positions
first_pos_dir = os.path.join(data_dirs[0], positions[0][0], 'Images')
first_pos_files = myutils.listdir(first_pos_dir)
channels, warn = cca_functions.find_available_channels(first_pos_files, first_pos_dir)

data_dirs, positions, file_names


# In[5]:


# MANUAL ENTRY OPTION: ALL OLD CAMERA DATA

data_dirs =  [
                'C:/Users/jyxiao/DATA/220401_JX_JX61a_oldcam', # WT
                'C:/Users/jyxiao/DATA/220407_SM_JX62b_oldcam', # 12A
                'C:/Users/jyxiao/DATA/220406_JX_JX63a_oldcam', # 7A
             ]

file_names = [
                '220401_JX_JX61a_oldcam',
                '220407_SM_JX62b_oldcam',
                '220406_JX_JX63a_oldcam',
             ]

positions = [
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6'],
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6'],
                ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6'],
            ]

image_folders = [[os.path.join(data_dir, pos_str, 'Images') for pos_str in pos_list] for pos_list, data_dir in zip(positions, data_dirs)]
# determine available channels based on first(!) position.
# Warn user if one or more of the channels are not available for some positions
first_pos_dir = os.path.join(data_dirs[0], positions[0][0], 'Images')
first_pos_files = myutils.listdir(first_pos_dir)
channels, warn = cca_functions.find_available_channels(first_pos_files, first_pos_dir)


# # Load data and perform calculations

# In[5]:


# overall_df is essentially the output from the ACDC GUI, plus some calculations for background
overall_df, is_timelapse_data, is_zstack_data = cca_functions.calculate_downstream_data(
    file_names,
    image_folders,
    positions,
    channels, 
    force_recalculation=False # if True, recalculates overall_df; use this if anything upstream is changed
)


# In[6]:


overall_df

# uncomment below to print all column headers in the dataframe
#for col in overall_df.columns:
#    print(col)

#overall_df_cell3 = overall_df[overall_df['Cell_ID'] == 3]
#print(overall_df_cell3)
#print(overall_df_cell3['Venus_corrected_amount'])


# In[7]:


frame_interval_minutes = 6 # frame interval of video

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
    overall_df_with_rel['time_in_phase'] = overall_df_with_rel['frame_i'] - overall_df_with_rel['phase_begin'] + 1 # in frames
    overall_df_with_rel['time_in_cell_cycle'] = overall_df_with_rel.groupby(['Cell_ID', 'generation_num', 'position', 'file'])['frame_i'].transform('cumcount') + 1 # in frames
    
    # calculated phase length in minutes
    overall_df_with_rel['phase_length_in_minutes'] = overall_df_with_rel.phase_length * frame_interval_minutes
    
    # calculate cycle length (only meaningful for mother S rows)
    overall_df_with_rel['cell_cycle_length_minutes'] = (overall_df_with_rel.time_in_cell_cycle - overall_df_with_rel.time_in_phase + overall_df_with_rel.phase_length)*frame_interval_minutes


# In[8]:


overall_df_with_rel


# In[9]:


# GENERATE FIELDS USEFUL FOR FACETING AND PLOTTING

# calculate a unique cell id across files by appending file, position, cell id, generation number (more like a cell cycle trace number, since we include generation number)
overall_df_with_rel['cell_unique_id'] = overall_df_with_rel.apply(
    lambda x: f'{x["file"]}_{x["position"]}_Cell_{x["Cell_ID"]}_Gen_{int(x["generation_num"])}', axis=1)

# determine which mutant by checking what substring filename contains
overall_df_with_rel['mutant'] = overall_df_with_rel.apply(
    lambda x:  'WT' if x.loc['file'].__contains__('JX61') \
        else ('12A' if x.loc['file'].__contains__('JX62') \
        else ( '7A' if x.loc['file'].__contains__('JX63') \
        else ( '5S' if x.loc['file'].__contains__('JX64') \
        else ('19A' if x.loc['file'].__contains__('JX68') \
        else ( '7T' if x.loc['file'].__contains__('JX69') \
        else ( 'Whi5 del' if x.loc['file'].__contains__('JX50') \
        else 'mutant error' ))))))
    ,axis=1)

# identify which camera by figuring out the date cutoff (April 15 2022)
overall_df_with_rel['camera'] = overall_df_with_rel.apply(
    lambda x:  'old' if int(x.loc['file'][0:6]) < 220415 
        else   'new' 
    ,axis=1)


# In[10]:


# AUTOFLUORESCENCE CORRECTIONS
ch_name = 'Venus' # name of channel of interest

# fit params below (a and b) obtained from autofluo_analysis.ipynb
# this is a per-pixel fit, which we determined was better than the whole-cell fit
# a is in units of au/pix/vox
# b is in units of au/pix
# cell area is always in pixels
# use vox to determine autofluo correction
# use fL for concentration calculation

a_new = 6.3695e-6
b_new = 7.70735
a_old = 1.4728361121456088e-05
b_old = 1.2569225809657871

# for old data, use ACDC correction since it's not flatfield-corrected ('corrected_amount' uses autoBkgr by default; could also change to dataPrepBkgr)
# for new data, use raw sum since it's flatfield-corrected

# main cell corrections
overall_df_with_rel[f'{ch_name}_af_corrected_amount'] = overall_df_with_rel.apply(
    lambda x: 
        x.loc[f'{ch_name}_corrected_amount'] - x.loc[f'cell_area_pxl']*(b_old + a_old*x.loc[f'cell_vol_vox']) if x.loc['camera'].__contains__('old') 
        else x.loc[f'{ch_name}_raw_sum'] - x.loc[f'cell_area_pxl']*(b_new + a_new*x.loc[f'cell_vol_vox'] )
    ,axis=1)
overall_df_with_rel[f'{ch_name}_af_corrected_concentration'] = overall_df_with_rel[f'{ch_name}_af_corrected_amount'] / overall_df_with_rel[f'cell_vol_fl'] # concentration is amount in au divided by vol in fL

# relative cell (i.e. associated bud) corrections
overall_df_with_rel[f'{ch_name}_af_corrected_amount_rel'] = overall_df_with_rel.apply(
    lambda x: 
        x.loc[f'{ch_name}_corrected_amount_rel'] - x.loc[f'area_rel']*(b_old + a_old*x.loc[f'cell_vol_vox_downstream_rel']) if x.loc['camera'].__contains__('old')
        else x.loc[f'{ch_name}_raw_sum_rel'] - x.loc[f'area_rel']*(b_new + a_new*x.loc[f'cell_vol_vox_downstream_rel'] )
    ,axis=1)
overall_df_with_rel[f'{ch_name}_af_corrected_concentration_rel'] = overall_df_with_rel[f'{ch_name}_af_corrected_amount_rel'] / overall_df_with_rel[f'cell_vol_fl_downstream_rel']


#for col in overall_df_with_rel.columns:
#    print(col)
    
overall_df_with_rel


# In[11]:



# GENERATE A NEW DATAFRAME FOR PLOTTING
# get indices of cells after filtering by criteria
filter_idx = (overall_df_with_rel['complete_cycle']==1)             &(overall_df_with_rel.generation_num==1) # can filter by anything, but for now just take complete cell cycles and first-generation mothers

# select needed cols from overall_df_with_rel
needed_cols = [
    'mutant', 'cell_unique_id', 'file', 'position', 'camera', # identifing info
    'cell_cycle_stage', 'frame_i', 'phase_length', 'phase_begin', 'time_in_phase', 'time_in_cell_cycle', 'cell_cycle_length_minutes',# time/age info
    'cell_vol_fl', 'cell_vol_fl_rel', 'cell_vol_vox', 'cell_vol_vox_rel', # size info
    'phase_volume_at_beginning', 'phase_volume_at_end', 'phase_combined_volume_at_end', # volume info
    f'{ch_name}_CV', f'{ch_name}_combined_amount_mother_bud',  f'{ch_name}_combined_raw_sum_mother_bud', # fluo info
    'phase_Venus_combined_amount_at_end',
    f'{ch_name}_af_corrected_amount', f'{ch_name}_af_corrected_amount_rel', f'{ch_name}_af_corrected_concentration', f'{ch_name}_af_corrected_concentration_rel', # fluo info
    'relationship', 'relative_ID' # relative info
]

# data to plot is a copy of the full dataframe, including only rows that meet the filter criterion and only taking columns needed for calculations/plotting
data_all = overall_df_with_rel.loc[filter_idx, needed_cols].copy()

# calculate the time the cell already spent in the current phase at the current timepoint
data_all['frames_in_phase'] = data_all['frame_i'] - data_all['phase_begin'] + 1
# calculate the time to the next (for G1 cells) and from the last (for S cells) G1/S transition  
data_all['bud_aligned_frames_in_phase'] = data_all.apply(
    lambda x: x.loc['frames_in_phase'] if x.loc['cell_cycle_stage']=='S' \
    else x.loc['frames_in_phase']-1-x.loc['phase_length'],
    axis=1
)

# calculate centered time in minutes
data_all['bud_aligned_time_in_minutes'] = data_all.bud_aligned_frames_in_phase * frame_interval_minutes

# calculate bud's contribution to total fluorescence
data_all['Bud amount'] = data_all.apply(
    lambda x: x.loc[f'{ch_name}_af_corrected_amount_rel'] if  x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' \
    else 0,
    axis=1
)
# calculate sum of mother and bud fluorescence
data_all['Combined m&b amount'] = data_all.apply(
    lambda x: x.loc[f'{ch_name}_af_corrected_amount'] + x.loc[f'{ch_name}_af_corrected_amount_rel'] if x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' \
    else x.loc[f'{ch_name}_af_corrected_amount'],
    axis=1
)
# calculate total volume of mother plus bud
data_all['Combined m&b volume fL'] = data_all.apply(
    lambda x: x.loc['cell_vol_fl']+x.loc['cell_vol_fl_rel'] if\
    x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' else\
    x.loc['cell_vol_fl'],
    axis=1
)

# calculate total volume of mother plus bud
data_all['Combined m&b volume vox'] = data_all.apply(
    lambda x: x.loc['cell_vol_vox']+x.loc['cell_vol_vox_rel'] if\
    x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' else\
    x.loc['cell_vol_vox'],
    axis=1
)

# final concentration is total Venus in mother/bud pair divided by their combined volume
data_all['Combined m&b concentration'] = data_all['Combined m&b amount']/data_all['Combined m&b volume fL']

# check that 'data_all' dataframe has necessary columns
data_all

# TO DO: write something that saves the processed data for plotting into a csv
# later, have a separate script (functions?) for loading csvs from different movies, to plot together


# In[ ]:





# In[ ]:


# NORMALIZING OLD DATA TO NEW DATA (PRESENTLY NOT USED)

# filter dataset for WT of one camera only, then get the "total fluorescence mean" on the af-corrected combined mother/bud amount
data_WT_new = data_all.loc[(data_all['mutant'] == 'WT') & (data_all['camera'] =='new')].copy()
data_WT_old = data_all.loc[(data_all['mutant'] == 'WT') & (data_all['camera'] =='old')].copy()

# first, get the norm factors for fluorescence amount and cell volume
fluo_mean_new = np.nanmean(data_WT_new['Combined m&b amount'])
fluo_mean_old = np.nanmean(data_WT_old['Combined m&b amount'])
fluo_scale_factor = fluo_mean_new/fluo_mean_old
vol_mean_new = np.nanmean(data_WT_new['Combined m&b volume fL'])
vol_mean_old = np.nanmean(data_WT_old['Combined m&b volume fL'])
vol_scale_factor = vol_mean_new/vol_mean_old

if np.isnan(fluo_scale_factor):
    fluo_scale_factor = 7.845085562993551

if np.isnan(vol_scale_factor):
    vol_scale_factor = 0.8520273998095671

print(f'{ch_name} scaling factor: ' + str(fluo_scale_factor))
print('Volume scaling factor: ' + str(vol_scale_factor))
print('Where scaling factor = new/old')

# normalize old camera data to new camera numbers
data_all['Combined m&b amount scaled'] = data_all.apply(
    lambda x: x.loc['Combined m&b amount'] * fluo_scale_factor if x.loc['camera'].__contains__('old')
        else  x.loc['Combined m&b amount']
    ,axis=1
)
data_all['Combined m&b volume scaled'] = data_all.apply(
    lambda x: x.loc['Combined m&b volume fL'] * vol_scale_factor if x.loc['camera'].__contains__('old')
        else  x.loc['Combined m&b volume fL']
    ,axis=1
)


# scale by mean of volume and fluo at budding
# filter for G1, WT strain
data_WT_new = data_all.loc[(data_all['mutant'] == 'WT') & (data_all['camera'] =='new') & (overall_df_with_rel['cell_cycle_stage'] == 'G1')].copy().drop_duplicates()
data_WT_old = data_all.loc[(data_all['mutant'] == 'WT') & (data_all['camera'] =='old') & (overall_df_with_rel['cell_cycle_stage'] == 'G1')].copy().drop_duplicates()

fluo_mean_new = np.nanmean(data_WT_new['phase_Venus_combined_amount_at_end'])
fluo_mean_old = np.nanmean(data_WT_old['phase_Venus_combined_amount_at_end'])
fluo_scale_factor2 = fluo_mean_new/fluo_mean_old


vol_mean_new = np.nanmean(data_WT_new['phase_volume_at_end'])
vol_mean_old = np.nanmean(data_WT_old['phase_volume_at_end'])
vol_scale_factor2 = vol_mean_new/vol_mean_old


if np.isnan(fluo_scale_factor2):
    fluo_scale_factor2 = 4.887952812741679

if np.isnan(vol_scale_factor2):
    vol_scale_factor2 = 0.806301635268053

print('Venus scaling factor 2: ' + str(fluo_scale_factor2))
print('Volume scaling factor 2: ' + str(vol_scale_factor2))
print('Where scaling factor = new/old')

data_all['Combined m&b amount scaled test'] = data_all.apply(
    lambda x: x.loc['Combined m&b amount'] * fluo_scale_factor2 if x.loc['camera'].__contains__('old')
        else  x.loc['Combined m&b amount']
    ,axis=1
)
data_all['Combined m&b volume scaled test'] = data_all.apply(
    lambda x: x.loc['Combined m&b volume fL'] * vol_scale_factor2 if x.loc['camera'].__contains__('old')
        else  x.loc['Combined m&b volume fL']
    ,axis=1
)


data_all['Combined m&b concentration scaled'] = data_all['Combined m&b amount scaled']/data_all['Combined m&b volume scaled']


# # Quantities over time, aligned to budding

# In[12]:


# choose fields to plot
x_var = 'bud_aligned_time_in_minutes'
y_var = 'Combined m&b concentration'
xlim = [-100,120]
ylim = [0,600]

# overlaying by mutant, position, etc.
hue = 'mutant'

# the line that does the actual plotting
FIG = sns.relplot(data=data_all, x=x_var, y=y_var, kind='line', hue=hue, height=8, aspect=1.2)#, legend='full', facet_kws={'legend_out': True})

# setting axis limits
FIG.set(xlim=xlim)
FIG.set(ylim=ylim)

# title and axis labels
FIG.fig.suptitle('CLN2pr-mVenus-PEST expression')
FIG.axes[0,0].set_xlabel('Time aligned to budding (min)')
#FIG.axes[0,0].set_ylabel('mVenus-PEST conc., m+b')

# generate legend  CURRENTLY NOT WORKING RIGHT
types = list(set(data_all[hue]))
num_types = len(types)
#leg_labels = list()
for i in range(0,num_types):
    data_temp = data_all.loc[data_all[hue]==types[i]].copy()
    numcells = len(set(data_temp['cell_unique_id']))
    print(types[i] + ', n = ' + str(numcells))
    #leg_labels.append(mutants[i] + ', n = ' + str(numcells))
    #FIG._legend.texts[i].set_text(mutants[i] + ', n = ' + str(numcells))


# In[13]:


# TEMP CELL

filter = (data_all['mutant'] != '5S')        &(data_all['mutant'] != '19A')        &(data_all['mutant'] != 'Whi5 del')

data_all_filt = data_all.loc[filter].copy()

# choose fields to plot
x_var = 'bud_aligned_time_in_minutes'
y_var = 'Combined m&b concentration'
xlim = [-100,120]
ylim = [0,600]

# overlaying by mutant, position, etc.
hue = 'mutant'

# the line that does the actual plotting
FIG = sns.relplot(data=data_all_filt, x=x_var, y=y_var, kind='line', hue=hue, height=8, aspect=1.2)#, legend='full', facet_kws={'legend_out': True})

# setting axis limits
FIG.set(xlim=xlim)
FIG.set(ylim=ylim)

# title and axis labels
FIG.fig.suptitle('CLN2pr-mVenus-PEST expression')
FIG.axes[0,0].set_xlabel('Time aligned to budding (min)')
#FIG.axes[0,0].set_ylabel('mVenus-PEST conc., m+b')

# generate legend  CURRENTLY NOT WORKING RIGHT
types = list(set(data_all_filt[hue]))
num_types = len(types)
#leg_labels = list()
for i in range(0,num_types):
    data_temp = data_all_filt.loc[data_all_filt[hue]==types[i]].copy()
    numcells = len(set(data_temp['cell_unique_id']))
    print(types[i] + ', n = ' + str(numcells))
    #leg_labels.append(mutants[i] + ', n = ' + str(numcells))
    #FIG._legend.texts[i].set_text(mutants[i] + ', n = ' + str(numcells))


# In[186]:


# if desired, plot one position at a time
mut_of_interest = '12A'

filter = (data_all['mutant'] == mut_of_interest)        &(data_all['file'] =='220610_JX_JX62c')         &(data_all['position'] == 'Position_4')             
data_solo = data_all.loc[filter].copy() # copy all fields in this case

#data_solo

numcells = len(set(data_solo['cell_unique_id'])) # number of gen-1 complete cell cycle traces

# facet by camera, experiment/file, position, etc.
#hue = 'position'
hue = 'cell_unique_id'
FIG_solo = sns.relplot(data = data_solo, x = x_var, y = y_var, kind = 'line', hue = hue, height = 8, aspect = 1.2, legend = 'full', facet_kws={'legend_out': True})


# setting axis limits
FIG_solo.set(xlim=xlim)
FIG_solo.set(ylim=ylim)

# title and axis labels
FIG_solo.fig.suptitle('Whi5[' + mut_of_interest + '], n = ' + str(numcells))
FIG_solo.axes[0,0].set_xlabel('Time aligned to budding (min)')
#FIG_solo.axes[0,0].set_ylabel('mVenus-PEST conc., m+b')

print('n = ' + str(numcells))


# In[40]:


# generate figures (aggregated, single traces, combined)
sns.set_theme(context='talk', font_scale=1.15)
sns.set_style("whitegrid", {"grid.color": ".95"})
fig, axs = plt.subplots(ncols=3, figsize=(30,10), sharey=False)
sns.lineplot(
    data=data_solo,
    x=x_var,
    y=y_var,
    ci=95,
    ax=axs[0]
)
sns.lineplot(
    data=data_solo,
    x=x_var, 
    y=y_var,
    estimator=None,
    units='cell_unique_id',
    ax=axs[1],
    lw=0.5,
    #alpha=0.5
)
sns.lineplot(
    data=data_solo,
    x=x_var, 
    y=y_var,
    ci=95,
    ax=axs[2]
)
sns.lineplot(
    data=data_solo,
    x=x_var, 
    y=y_var,
    estimator=None,
    units='cell_unique_id',
    ax=axs[2],
    lw=0.5,
    #alpha=0.5
)
fig.suptitle('Cln2pr expression of Whi5[' + mut_of_interest + '], n = ' + str(numcells))
axs[0].set_xlim(xlim)
#axs[0].set_ylim(ylim)
#axs[1].set_xlim(xlim)
#axs[1].set_ylim(ylim)
#axs[2].set_xlim(xlim)
#axs[2].set_ylim(ylim)

#plt.savefig('../figures/firstgen_g1_concentration.png', dpi=300)
plt.show()


# # Distributions at birth, budding, or division

# In[139]:



# G1 length distribution: histogram plus cumulative versions

# kdeplot or hisplot from seaborn
# use drop duplicates function to trim redundancy

#mut_of_interest = 'Whi5 del'

# GENERATE A NEW DATAFRAME FOR PLOTTING
filter_idx = (overall_df_with_rel['complete_cycle']==1)             &(overall_df_with_rel.generation_num==1)             &(overall_df_with_rel['cell_cycle_stage'] == 'G1')#            &(overall_df_with_rel['file'] == '220516_JX_JX63b')\
#            &(overall_df_with_rel['mutant'] == mut_of_interest)

# select needed cols from overall_df_with_rel; for this part, make sure to only take fields that are constant for a given cell cycle
needed_cols = [
    'mutant', 'cell_unique_id', 'file', 'position', 'camera', # identifing info
    'cell_cycle_stage', 'phase_length', 'phase_length_in_minutes', 'cell_cycle_length_minutes', # time/age info
    'phase_volume_at_beginning', 'phase_volume_at_end', 'phase_combined_volume_at_end', # volume info
    'phase_'f'{ch_name}_amount_at_beginning', 'phase_'f'{ch_name}_combined_amount_at_beginning', 'phase_'f'{ch_name}_combined_amount_at_end', # fluo info
    'phase_'f'{ch_name}_concentration_at_beginning', 'phase_'f'{ch_name}_concentration_at_end'
]

# data to plot is a copy of the full dataframe, including only rows that meet the filter criterion and only taking columns needed for calculations/plotting
data_all_nodups = overall_df_with_rel.loc[filter_idx, needed_cols].copy().drop_duplicates()

#x_var = 'phase_volume_at_beginning'
#x_var = 'phase_combined_volume_at_end'
x_var = 'phase_length_in_minutes'
#x_var = 'cell_cycle_length_minutes'

hue = 'file'

sns.set(rc={'figure.figsize':(11.7,8.27)})

FIG_CDF = sns.ecdfplot(data=data_all_nodups, x=x_var, hue=hue)

#FIG_CDF.axes.set_xlabel('Volume at birth (fL)')
#FIG_CDF.axes.set_xlabel('Volume at budding (fL)')
FIG_CDF.axes.set_xlabel('G1 length (minutes)')
#FIG_CDF.axes.set_xlabel('S/G2/M length (minutes)')
#FIG_CDF.axes.set_xlabel('Full cycle length (minutes)')

FIG_CDF.axes.set_xlim([0,None])

# generate legend  CURRENTLY NOT WORKING RIGHT
types = list(set(data_all_nodups[hue]))
num_types = len(types)
#leg_labels = list()
for i in range(0,num_types):
    data_temp = data_all.loc[data_all[hue]==types[i]].copy()
    numcells = len(set(data_temp['cell_unique_id']))
    print(types[i] + ', n = ' + str(numcells))
    #leg_labels.append(mutants[i] + ', n = ' + str(numcells))
    #FIG._legend.texts[i].set_text(mutants[i] + ', n = ' + str(numcells))

#sns.lineplot(data=data_nodups, x = "phase_volume_at_beginning", y = "phase_volume_at_end", hue=hue, kind='scatter')


# In[ ]:





# In[86]:


# TO DO: add a relplot line that plots each position for a given experiment
# then, add another relplot line that breaks a given experiment into constituent positions
data_all_nodups['phase_volume_at_beginning']


# In[198]:


# Beno's code that plots histogram of overall cycle length; only needs overall_df_with_rel

complete_cc_data = overall_df_with_rel[overall_df_with_rel.complete_cycle==1]
cc_lengths = complete_cc_data.groupby(['Cell_ID', 'generation_num', 'file', 'position'])['time_in_cell_cycle'].max() * frame_interval_minutes
sns.ecdfplot(cc_lengths,hue='mutant')
plt.show()


# In[ ]:





# In[ ]:





# In[77]:


# OLD AUTOFLUORESCENCE CALCULATIONS BLOCK (UNUSED NOW)

# name of channel to be plotted
ch_name = 'Venus'
frame_interval_minutes = 6 # frame interval of video
camera = 'old'

# load fields needed for calculations
cell_area = overall_df_with_rel[f'cell_area_pxl']
cell_vol_vox = overall_df_with_rel[f'cell_vol_vox'] # use vox to determine autofluo correction
cell_vol_fl = overall_df_with_rel[f'cell_vol_fl'] # use fL for concentration calculation

cell_area_rel = overall_df_with_rel[f'area_rel']
cell_vol_vox_rel = overall_df_with_rel[f'cell_vol_vox_downstream_rel'] # use vox to determine autofluo correction
cell_vol_fl_rel = overall_df_with_rel[f'cell_vol_fl_downstream_rel'] # use fL for concentration calculation

if camera == 'new': # anything after/including April 15 2022
    a = 6.3695e-6
    b = 7.70735
    
    fluo_tot = overall_df_with_rel[f'{ch_name}_raw_sum'] # use raw sum because we do a flat field correction beforehand, so ACDC one is unnecessary/redundant
    fluo_tot_rel = overall_df_with_rel[f'{ch_name}_raw_sum_rel']
    
elif camera == 'old': # anything before April 15 2022
    a = 1.4728361121456088e-05
    b = 1.2569225809657871
    
    fluo_tot = overall_df_with_rel[f'{ch_name}_corrected_amount'] # CONSIDER CHANGING THIS TO {ch_name}_amount_dataPrepBkgr; by default, corrected_amount uses autoBkgr
    fluo_tot_rel = overall_df_with_rel[f'{ch_name}_corrected_amount_rel']
    
else:
    print('ERROR: camera not specified')

autofluo_correction = cell_area*(b + a*cell_vol_vox)
autofluo_correction_rel = cell_area_rel*(b + a*cell_vol_vox_rel)

# main cell correction
overall_df_with_rel[f'{ch_name}_af_corrected_amount'] = fluo_tot - autofluo_correction
overall_df_with_rel[f'{ch_name}_af_corrected_concentration'] = overall_df_with_rel[f'{ch_name}_af_corrected_amount'] / cell_vol_fl # concentration is amount in au divided by vol in fL
# relative cell (associated bud) correction
overall_df_with_rel[f'{ch_name}_af_corrected_amount_rel'] = fluo_tot_rel - autofluo_correction_rel
overall_df_with_rel[f'{ch_name}_af_corrected_concentration_rel'] = overall_df_with_rel[f'{ch_name}_af_corrected_amount_rel'] / cell_vol_fl_rel # concentration is amount in au divided by vol in fL

#overall_df_with_rel[f'{ch_name}_af_corrected_amount_rel_TEST'] - overall_df_with_rel[f'{ch_name}_af_corrected_amount_rel']


# In[ ]:





# In[ ]:





# In[ ]:





# In[64]:


split_by_gen = False
min_no_of_ccs = 0

if split_by_gen:
    g_cols = ['bud_aligned_frames_in_phase', 'Generation']
else:
    g_cols = 'bud_aligned_frames_in_phase'
    
# group dataframe to calculate sample sizes per generation
standard_grouped = data_all.groupby(  ['position', 'file', 'Cell_ID', 'generation_num']  ).agg('count').reset_index()
data_all['Generation'] = data_all.apply(
    lambda x: f'1st ($n_1$={len(standard_grouped[standard_grouped.generation_num==1])})' if\
    x.loc['generation_num']==1 else f'2+ ($n_2$={len(standard_grouped[standard_grouped.generation_num>1])})',
    axis=1
)
data_all['contributing_ccs_at_time'] = data_all.groupby(g_cols).transform('count')['selection_subset']
data_all = data_all[data_all.contributing_ccs_at_time >= min_no_of_ccs]

# finally prepare data for plot (use melt for multiple lines)
sample_size = len(standard_grouped)
avg_cell_cycle_length = round(standard_grouped.loc[:,'bud_aligned_time_in_minutes'].mean())*frame_interval_minutes
cols_to_plot = ['Combined m&b concentration']
index_cols = [col for col in data_all.columns if col not in cols_to_plot]

data_melted = pd.melt(
    data_all, index_cols, var_name='Method of calculation'
).sort_values('Method of calculation')
#data_dir = os.path.join('..', 'data', 'paper_plot_data')
# save preprocessed data
#data.to_csv(os.path.join(data_dir, 'data.csv'), index=False)
#data.to_csv(os.path.join(data_dir, 'data.csv'), index=False)

#data_melted


# In[16]:


# plot the data, comment out "style" argument to not make difference between generations
sns.set_theme(style="darkgrid", font_scale=1.6)
f, ax = plt.subplots(figsize=(15, 12))
if split_by_gen:
    style='Generation'
else:
    style=None
ax = sns.lineplot(
    data=data_melted,#.sort_values('Pool, Phase'),
    x="bud_aligned_time_in_minutes", 
    y="value",
    hue="cell_unique_id", # NOTE: uncommenting this separates the data by the column given, in this case position (so each position plotted separately)
    style=style,
    ci=95
)
ax.axvline(x=0, color='red', label='Time of Bud Emergence')
ax.text(
    0.5, 0.21, "Time of \nBud Emergence", position=(2,5), #horizontalalignment='center', verticalalignment='top', 
    size='medium', color='red', weight='normal'
)
ax.legend(
    title=f'Avg CC Length: {avg_cell_cycle_length} min, n = {sample_size}', 
    fancybox=True,
    labelspacing=0.5,
    handlelength=1.5,
    loc = 'upper left'
)
ax.set_ylabel("Venus concentration (mother/bud combined) (a.u.)", fontsize=20)
ax.set_xlabel("Time (budding at 0) (minutes)", fontsize=20)
ax.set_title("Cln2pr-mVenus-PEST", fontsize=30)
#plt.tight_layout()
plt.show()


# # Plot gallery - timelapse data

# ## (Volume) growth in G1 vs. mother+daughter growth in S (1st generation)

# In[18]:


# obtain table where one cell cycle is represented by one row: 
# first set of columns (like phase_length, growth...) for G1, second set of cols for S
complete_cc_data = phase_grouped[phase_grouped.all_complete==1]
s_data = complete_cc_data[complete_cc_data.cell_cycle_stage=="S"]
g1_data = complete_cc_data[complete_cc_data.cell_cycle_stage=="G1"]
plot_data2 = g1_data.merge(
    s_data, on=['Cell_ID', 'generation_num', 'position'], how='inner', suffixes=('_g1','_s')
)
plot_data2 = plot_data2[plot_data2.generation_num==1]
plot_data2['combined_motherbud_growth'] = plot_data2['phase_area_growth_s'] + plot_data2['phase_daughter_area_growth_s']
plot_data2['combined_motherbud_vol_growth'] = plot_data2['phase_volume_growth_s'] + plot_data2['phase_daughter_volume_growth_s']


# In[21]:


sns.set_theme(style="darkgrid", font_scale=2)
# Initialize the figure
g = sns.lmplot(x="phase_volume_growth_g1", y="combined_motherbud_vol_growth", data=plot_data2,
    hue="selection_subset_g1", height=10)
g._legend.set_title('Position Pool')
ax = plt.gca()
ax.set_ylabel("Combined Mother+Bud S growth [fL]", fontsize=20)
ax.set_xlabel("G1 growth [fL]", fontsize=20)
ax.set_title("G1 volume growth vs. mother+daughter cell growth in S phase", fontsize=30)
plt.show()


# ## Volume at birth vs. G1 duration (1st generation)
# - Plot to determine if there is a negative correlation between cell size at birth and length of the first G1 phase

# In[22]:


# obtain table where one cell cycle is represented by one row: 
# first set of columns (like phase_length, growth...) for G1, second set of cols for S
plot_data3 = phase_grouped[phase_grouped.cell_cycle_stage=="G1"]
plot_data3 = plot_data3[plot_data3.complete_phase==1]
plot_data3 = plot_data3[plot_data3.generation_num==1]

sns.set_theme(style="darkgrid", font_scale=2)
# Initialize the figure
g = sns.lmplot(x="phase_volume_at_beginning", y="phase_length", data=plot_data3,
    hue="selection_subset", height=10)
g._legend.set_title('Position Pool')
ax = plt.gca()
ax.set_ylabel("Duration of first G1 phase [no of frames]", fontsize=20)
ax.set_xlabel("Volume at birth (first cytokinesis) [fL]", fontsize=20)
ax.set_title("Volume at birth vs G1 duration (1st generation)", fontsize=30)
plt.show()


# ## Volume at birth vs. Signal concentration at birth (1st generation)

# In[23]:


# set channel name here:
ch_name = 'Venus'
# obtain table where one cell cycle is represented by one row: 
# first set of columns (like phase_length, growth...) for G1, second set of cols for S
plot_data4 = phase_grouped[phase_grouped.cell_cycle_stage=="G1"]
plot_data4 = plot_data4[plot_data4.complete_phase==1]
plot_data4 = plot_data4[plot_data4.generation_num==1]

sns.set_theme(style="darkgrid", font_scale=2)
# Initialize the figure
g = sns.lmplot(x="phase_volume_at_beginning", y=f"phase_{ch_name}_concentration_at_beginning", data=plot_data4,
    hue="selection_subset", height=10, )
g._legend.set_title('Position Pool')
g.set(yscale="log")
ax = plt.gca()
ax.set_ylabel("Venus signal amount per volume in cell [a.u.]", fontsize=20)
ax.set_xlabel("Volume at birth (first cytokinesis) [fL]", fontsize=20)
ax.set_title("Volume at birth vs Venus signal amount per volume (1st generation)", fontsize=30)
plt.show()


# ## G1 vs. S duration (1st generation)

# In[24]:


# obtain table where one cell cycle is represented by one row: 
# first set of columns (like phase_length, growth...) for G1, second set of cols for S
complete_cc_data = phase_grouped[phase_grouped.all_complete==1]
s_data = complete_cc_data[complete_cc_data.cell_cycle_stage=="S"]
g1_data = complete_cc_data[complete_cc_data.cell_cycle_stage=="G1"]
plot_data1 = g1_data.merge(s_data, on=['Cell_ID', 'generation_num', 'position', 'file'], how='inner')
plot_data1 = plot_data1[plot_data1.generation_num==1]

sns.set_theme(style="darkgrid", font_scale=2)
# Initialize the figure
g = sns.lmplot(x="phase_length_x", y="phase_length_y", data=plot_data1,
    hue="selection_subset_x", height=10)
g._legend.set_title('Position Pool')
ax = plt.gca()
ax.set_ylabel("S duration same cycle [frames]", fontsize=20)
ax.set_xlabel("G1 duration [frames]", fontsize=20)
ax.set_title("G1 duration vs. S duration within same generation", fontsize=30)
plt.show()


# In[66]:


complete_cc_data


# ## Distribution of Cell volumes

# In[25]:


sns.set_theme(style="ticks", font_scale=2)

# Initialize the figure
plt.figure(figsize=(10,10))
sns.histplot(
    x='cell_vol_fl', 
    data=overall_df,
    hue='relationship',
    bins=20,
    legend=False
)
ax = plt.gca()
labels = [
    'Mother cells',
    'Buds'
]
handles = [
    mpatches.Patch(color=sns.color_palette('pastel')[0]),
    mpatches.Patch(color=sns.color_palette('pastel')[1])
]
ax.legend(
    handles=handles,
    labels=labels, 
    loc='upper right',
    #bbox_to_anchor = (1,0.2),
    framealpha=0.5
)

# Tweak the visual presentation
ax = plt.gca()
ax.set_xlabel("Cell volume [fL]", fontsize=20)
ax.set_title(f"Volume distribution, n: {overall_df.shape[0]}", fontsize=30)
#sns.despine(trim=True, left=True)
plt.show()


# # ACDC paper figures

# ## Flurescence Signal over time (centered on bud emergence)
# - timelapse data is assumed here
# - note that in this plot, we filter for selection_subset==0
# - make sure to change this if unwanted or to select the data with flu signal to be in the first pool

# In[78]:


# some configurations
# frame interval of video
frame_interval_minutes = 6
# quantiles of complete cell cycles (wrt phase lengths) to exclude from analysis 
# (not used, keep this for potential later use)
down_q, upper_q = 0, 1
# minimum number of cell cycles contributing to the mean+CI curve:
min_no_of_ccs = 0
# determine if you want to split the plot by generation
split_by_gen = False
# whether to scale to 0/1 or not
scale_data = False
# name of channel the signal of which should be plotted
ch_name = 'Venus'


# In[79]:


# select needed cols from overall_df_with_rel to not end up with too many columns
needed_cols = [
    'selection_subset', 'position', 'Cell_ID', 'cell_cycle_stage', 'generation_num', 'frame_i',
    f'{ch_name}_corrected_amount', f'{ch_name}_corrected_amount_rel', f'{ch_name}_corrected_concentration', f'{ch_name}_corrected_concentration_rel', 
    'file', 'relationship', 'relative_ID', 'phase_length', 'phase_begin']#, f'gui_{ch_name}_amount_autoBkgr'
#]
filter_idx = np.logical_and(overall_df_with_rel['complete_cycle'] == 1, overall_df_with_rel.selection_subset==0, overall_df_with_rel.generation_num==1)
plot_data4a = overall_df_with_rel.loc[filter_idx, needed_cols].copy()
# calculate the time the cell already spent in the current frame at the current timepoint
plot_data4a['frames_in_phase'] = plot_data4a['frame_i'] - plot_data4a['phase_begin'] + 1
# calculate the time to the next (for G1 cells) and from the last (for S cells) G1/S transition  
plot_data4a['centered_frames_in_phase'] = plot_data4a.apply(
    lambda x: x.loc['frames_in_phase'] if\
    x.loc['cell_cycle_stage']=='S' else\
    x.loc['frames_in_phase']-1-x.loc['phase_length'],
    axis=1
)
# calculate combined signal and the "Pool, Phase ID" for the legend
# plot_data4a at this point only contains relationship==mother, 
# as generation_num==0 and relationship==bud are filtered out (incomplete cycle, cycles start with G1)
plot_data4a['Combined signal m&b'] = plot_data4a.apply(
    lambda x: x.loc[f'{ch_name}_corrected_amount']+x.loc[f'{ch_name}_corrected_amount_rel'] if\
    x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' else\
    x.loc[f'{ch_name}_corrected_amount'],
    axis=1
)
plot_data4a['Bud signal'] = plot_data4a.apply(
    lambda x: x.loc[f'{ch_name}_corrected_amount_rel'] if\
    x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' else 0,
    axis=1
)
# scale data if needed
if scale_data:
    maximum = max(
        plot_data4a['Combined signal m&b'].max(), 
        plot_data4a['Bud signal'].max()
    )
    plot_data4a['Combined signal m&b'] /= maximum
    plot_data4a['Bud signal'] /= maximum
# calculate min and max centered times per generation to eliminate up to a percentile
# (not used, as upper_q and lower_q are set to 100/0 respectively)
plot_data4a['min_centered_frames'] = plot_data4a.groupby(
    ['position', 'file', 'Cell_ID', 'generation_num']
)['centered_frames_in_phase'].transform(
    'min'
)
plot_data4a['max_centered_frames'] = plot_data4a.groupby(
    ['position', 'file', 'Cell_ID', 'generation_num']
)['centered_frames_in_phase'].transform(
    'max'
)
min_and_max = plot_data4a.groupby(
    ['Cell_ID', 'generation_num', 'position', 'file']
).agg(
    min_centered = ('min_centered_frames', 'first'),
    max_centered = ('max_centered_frames', 'first')
).reset_index()
min_val, max_val = np.quantile(
    min_and_max.min_centered, down_q
) * frame_interval_minutes, np.quantile(
    min_and_max.max_centered, upper_q
) * frame_interval_minutes
# perform selection (won't change anything if upper and lower are 100 and 0 respectively)
selection_indices = np.logical_and(
    plot_data4a.min_centered_frames*frame_interval_minutes>=min_val, 
    plot_data4a.max_centered_frames*frame_interval_minutes<=max_val
)
plot_data4a = plot_data4a[selection_indices]

# calculate centered time in minutes
plot_data4a['centered_time_in_minutes'] = plot_data4a.centered_frames_in_phase * frame_interval_minutes

# group dataframe to calculate sample sizes per generation
standard_grouped = plot_data4a.groupby(
    ['position', 'file', 'Cell_ID', 'generation_num']
).agg('count').reset_index()
plot_data4a['Generation'] = plot_data4a.apply(
    lambda x: f'1st ($n_1$={len(standard_grouped[standard_grouped.generation_num==1])})' if\
    x.loc['generation_num']==1 else f'2+ ($n_2$={len(standard_grouped[standard_grouped.generation_num>1])})',
    axis=1
)
if split_by_gen:
    g_cols = ['centered_frames_in_phase', 'Generation']
else:
    g_cols = 'centered_frames_in_phase'
plot_data4a['contributing_ccs_at_time'] = plot_data4a.groupby(g_cols).transform('count')['selection_subset']
plot_data4a = plot_data4a[plot_data4a.contributing_ccs_at_time >= min_no_of_ccs]

# finally prepare data for plot (use melt for multiple lines)
sample_size_4a = len(standard_grouped)
avg_cell_cycle_length = round(standard_grouped.loc[:,'centered_time_in_minutes'].mean())*frame_interval_minutes
cols_to_plot = ['Bud signal', 'Combined signal m&b']
index_cols = [col for col in plot_data4a.columns if col not in cols_to_plot]
plot_data4a_melted = pd.melt(
    plot_data4a, index_cols, var_name='Method of calculation'
).sort_values('Method of calculation')
data_dir = os.path.join('..', 'data', 'paper_plot_data')
# save preprocessed data for Fig. 4A
#plot_data4a_melted.to_csv(os.path.join(data_dir, 'plot_data4a_melted.csv'), index=False)
#plot_data4a.to_csv(os.path.join(data_dir, 'plot_data4a.csv'), index=False)


# In[80]:


# plot the data, comment out "style" argument to not make difference between generations
sns.set_theme(style="darkgrid", font_scale=1.6)
f, ax = plt.subplots(figsize=(15, 12))
if split_by_gen:
    style='Generation'
else:
    style=None
ax = sns.lineplot(
    data=plot_data4a_melted,#.sort_values('Pool, Phase'),
    x="centered_time_in_minutes", 
    y="value",
    #hue='Cell_ID',
    #hue='position', # NOTE: uncommenting this separates the data by the column given, in this case position (so each position plotted separately)
    style=style,
    ci=95
)
ax.axvline(x=0, color='red')#, label='Time of Bud Emergence')
ax.text(
    0.5, 0.21, "Time of \nBud Emergence", position=(2,100), #horizontalalignment='center', verticalalignment='top', 
    size='medium', color='red', weight='normal'
)
ax.legend(
    title=f'Avg CC Length: {avg_cell_cycle_length} min, n = {sample_size_4a}', 
    fancybox=True,
    labelspacing=0.5,
    handlelength=1.5,
    loc = 'upper left'
)
ax.set_ylabel("Total Venus amount corrected by background [a.u.]", fontsize=20)
ax.set_xlabel("Time in phase relative to G1/S transition [minutes]", fontsize=20)
ax.set_title("JX62b: Whi5-12A, Cln2pr-mVenus-PEST", fontsize=30)
plt.tight_layout()
plt.show()


# ## Volume at birth and division vs. mCitrine amount at birth (single cell) and division (combined)
# - This plot is based on the grouped-by-phase data
# - We assume that cell cycle phases were annotated with ACDC (~ column "cell_cycle_stage" exists, relative's dataframe was calculated and attached)

# In[26]:


# configure channel the signal of which should be plotted
ch_name = 'Venus'
# first set of columns (like phase_length, growth...) for G1, second set of cols for S
needed_cols = [
    'Cell_ID', 'generation_num', 'position', 'file', 'cell_cycle_stage', 'selection_subset', 
    'phase_volume_at_beginning', 'phase_volume_at_end', f'phase_{ch_name}_amount_at_beginning',
    f'phase_{ch_name}_combined_amount_at_end','phase_combined_volume_at_end'
]
plot_data4 = phase_grouped.loc[phase_grouped.complete_cycle==1, needed_cols]
scale_data = False


# In[27]:


plot_data4['relevant_volume'] = plot_data4.apply(
    lambda x: x.loc['phase_volume_at_beginning'] if\
    x.loc['cell_cycle_stage']=='G1' else\
    x.loc['phase_combined_volume_at_end'],
    axis=1
)
plot_data4['relevant_amount'] = plot_data4.apply(
    lambda x: x.loc[f'phase_{ch_name}_amount_at_beginning'] if\
    x.loc['cell_cycle_stage']=='G1' else\
    x.loc[f'phase_{ch_name}_combined_amount_at_end'],
    axis=1
)
# defining a function to generate entries for the figure legend 
# (assuming that selection_subset>0 is the autofluorescence control of the experiment)
def calc_legend_entry(x):
    if x.loc['selection_subset'] == 0:
        if x.loc['cell_cycle_stage']=='G1':
            return 'Single cell at birth'
        else:
            return 'Combined mother&bud at cytokinesis'
    else:
        if x.loc['cell_cycle_stage']=='G1':
            return 'Af control, single cell at birth'
        else:
            return 'Af control, combined mother&bud at cytokinesis'
        
plot_data4['Kind of Measurement'] = plot_data4.apply(
    lambda x: 'Single Cell in G1 (Frame after Cytokinesis)' if\
    x.loc['cell_cycle_stage']=='G1' else\
    'Combined Mother & Bud in S (Frame before Cytokinesis)',
    axis=1
)
plot_data4['Kind of Measurement new'] = plot_data4.apply(
    calc_legend_entry,
    axis=1
)
plot_data4['Generation'] = plot_data4.apply(
    lambda x: f'1st ($n_1$={int(len(plot_data4[plot_data4.generation_num==1])/2)})' if\
    x.loc['generation_num']==1 else f'2+ ($n_2$={int(len(plot_data4[plot_data4.generation_num>1])/2)})',
    axis=1
)
if scale_data:
    maximum = plot_data4['relevant_amount'].max()
    plot_data4['relevant_amount'] /= maximum
sample_size = len(plot_data4)


# In[28]:


#plot_data4 = plot_data4[plot_data4.selection_subset==1]
sns.set_theme(style="darkgrid", font_scale=1.6)
# create lmplot. Don't scatter and ommit legend to customize scatterplot and legend
sns.lmplot(
    x="relevant_volume", 
    y="relevant_amount", 
    data=plot_data4.sort_values(
        'Kind of Measurement new', ascending=False
    ),
    hue="Kind of Measurement new",
    legend=False,
    height=10,
    aspect=1.1,
    scatter=False
)
sns.scatterplot(
    x="relevant_volume", 
    y="relevant_amount", 
    data=plot_data4[plot_data4.generation_num==1].sort_values(
        'Kind of Measurement new', ascending=False
    ),
    hue="Kind of Measurement new",
    legend=False,
    marker='x'
)
sns.scatterplot(
    x="relevant_volume", 
    y="relevant_amount", 
    data=plot_data4[plot_data4.generation_num>1].sort_values(
        'Kind of Measurement new', ascending=False
    ),
    hue="Kind of Measurement new",
    legend=False,
    marker='o'
)
ax = plt.gca()
labels = [
    'Single cell at birth',
    'Combined mother&bud at cytokinesis',
    'Af control, single cell at birth',
    'Af control, combined mother&bud at cytokinesis',
    'Generation 1',
    'Generation 2+'
]
handles = [
    mpatches.Patch(color=sns.color_palette()[0]),
    mpatches.Patch(color=sns.color_palette()[1]),
    mpatches.Patch(color=sns.color_palette()[2]),
    mpatches.Patch(color=sns.color_palette()[3]),
    mlines.Line2D([], [], color='gray', marker='x', linestyle='None',
                          markersize=10),
    mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                          markersize=10)
]
ax.legend(
    handles=handles,
    labels=labels, 
    loc='center right',
    bbox_to_anchor = (1,0.2),
    framealpha=0.5
)
ax.set_ylabel("Amount of Signal in Cell(s) [a.u.]", fontsize=20)
ax.set_xlabel("Volume at Birth / Combined Volume Before Cytokinesis [fL]", fontsize=20)
ax.set_title(f"Volume at birth vs Signal Amount (n={int(sample_size/2)})", fontsize=30)
# format y-axis
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
ax.get_yaxis().get_offset_text().set_position((-0.05,0))
# format x-axis
ax.set_xlim(0, plot_data4.relevant_volume.max()+20)
plt.tight_layout()
plt.show()
print(f'sample size flu-control: {len(plot_data4[plot_data4.selection_subset==1])//2}')
print(f'sample size tagged strain: {len(plot_data4[plot_data4.selection_subset==0])//2}')


# In[ ]:




