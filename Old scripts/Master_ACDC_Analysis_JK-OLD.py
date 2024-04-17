#!/usr/bin/env python
# coding: utf-8

#%%

import os
import sys
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
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import random
ch_name = 'mCitrine'
#%%


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Configurations
# - follow the file selection dialog:
#     - select microscopy folder in first step
#     - select positions of the selected folder in second step
# - repeat to add more positions to the analysis
# - positions selected within one iteration of the dialog will be pooled together in the following analyses

#%% choose expt/positions you want to load


data_dirs, positions = cca_functions.configuration_dialog()
file_names = [os.path.split(path)[-1] for path in data_dirs]
image_folders = [[os.path.join(data_dir, pos_str, 'Images') for pos_str in pos_list] for pos_list, data_dir in zip(positions, data_dirs)]
# determine available channels based on first(!) position.
# Warn user if one or more of the channels are not available for some positions
first_pos_dir = os.path.join(data_dirs[0], positions[0][0], 'Images')
first_pos_files = myutils.listdir(first_pos_dir)
channels, warn = cca_functions.find_available_channels(first_pos_files, first_pos_dir)

data_dirs, positions, file_names


# # Load data and perform calculations

#%% takes acdc output from each indicated position from the previous file and creates a dataframe (overall_df) with all the measurements


overall_df, is_timelapse_data, is_zstack_data = cca_functions.calculate_downstream_data(
    file_names,
    image_folders,
    positions,
    channels, 
    force_recalculation=False # if True, recalculates overall_df; use this if anything upstream is changed
)


# In[228]:


# write something that checks the columns to see if they exist before adding these

overall_df["Strain"] = 'yJK098'
overall_df["aTc"] = '30ng/mL'
overall_df["Date"] = '20220523'
overall_df["Replicate"] = 'Replicate 2'



# In[229]:


# if cell cycle annotations were performed in ACDC (i.e. this is a timecourse movie), extend the dataframe by a join on each cell's relative cell
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


# In[ ]:





# In[230]:


# name of channel the signal of which should be plotted
ch_name = 'mCitrine'

# AUTOFLUORESCENCE CALCULATIONS for the channel named above

# load fields from overall dataframe with relatives appended
cell_area = overall_df_with_rel[f'cell_area_pxl']
cell_vol_vox = overall_df_with_rel[f'cell_vol_vox'] # use vox to determine autofluo correction
cell_vol_fl = overall_df_with_rel[f'cell_vol_fl'] # use fL for concentration calculation
#Ven_tot = overall_df_with_rel[f'{ch_name}_corrected_amount'] # CONSIDER CHANGING THIS TO Venus_amount_dataPrepBkgr
Ven_tot = overall_df_with_rel[f'{ch_name}_sum']
cell_area_rel = overall_df_with_rel[f'area_rel']
cell_vol_vox_rel = overall_df_with_rel[f'cell_vol_vox_downstream_rel'] # use vox to determine autofluo correction
cell_vol_fl_rel = overall_df_with_rel[f'cell_vol_fl_downstream_rel'] # use fL for concentration calculation
Ven_tot_rel = overall_df_with_rel[f'{ch_name}_sum_rel']
#Ven_tot_rel = overall_df_with_rel[f'{ch_name}_corrected_amount_rel']

# AF PER PIXEL FIT (ROBUST), gating 0 voxels
#a = 1.4728361121456088e-05 # units au/pix/vox
#b = 1.2569225809657871 # units au/pix

a = 6.369499109612808e-6
b = 7.707351747330752

autofluo_correction = cell_area*(b + a*cell_vol_vox)
autofluo_correction_rel = cell_area_rel*(b + a*cell_vol_vox_rel)

# TOTAL AF FIT (ROBUST), gating 3000 voxels
#a = 0.03642486864549634
#b = 6.690903966826931
#autofluo_correction = b + a*cell_vol_vox
#autofluo_correction_rel = b + a*cell_vol_vox_rel

# concentration is amount in au divided by vol in fL
# main cell correction
overall_df_with_rel[f'{ch_name}_af_corrected_amount'] = Ven_tot - autofluo_correction
overall_df_with_rel[f'{ch_name}_af_corrected_concentration'] = overall_df_with_rel[f'{ch_name}_af_corrected_amount'] / cell_vol_fl

# relative cell (associated bud) correction
overall_df_with_rel[f'{ch_name}_af_corrected_amount_rel'] = Ven_tot_rel - autofluo_correction_rel
overall_df_with_rel[f'{ch_name}_af_corrected_concentration_rel'] = overall_df_with_rel[f'{ch_name}_af_corrected_amount_rel'] / cell_vol_fl_rel


#print(overall_df_with_rel[f'Venus_af_corrected_concentration'])


# In[231]:


# calculate a unique cell id across files by appending file, position, cell id, generation number
overall_df_with_rel['cell_unique_id'] = overall_df_with_rel.apply(lambda x: f'{x["file"]}_{x["position"]}_Cell_{x["Cell_ID"]}_Gen_{int(x["generation_num"])}', axis=1)


# In[21]:


overall_df_with_rel[overall_df_with_rel['generation_num'].isna()]['position']


# In[232]:


#for col in overall_df_with_rel.columns:
#    print(col)

# @todo: add column filter step

#overall_df_with_rel["mCitrine_sum"]
#autofluo_correction
#overall_df_with_rel.groupby('cell_unique_id')
#print(grouped)
#overall_df_with_rel['position']
csv_path = r'C:\Users\Jacob Kim\Google Drive (jmhkim@stanford.edu)\ffc_test\bulktest\acdc_files\220603_yJK100_30ngmLaTc_expt1_data\output_df.csv'
overall_df_with_rel.to_csv(path_or_buf = csv_path,index=False)
#overall_df_with_rel.to_clipboard()

# @todo: add step where e.g. first gen daughter cells are exported to their own .csv file (for sharing w colleagues)


# In[233]:

    # @todo: 2____ this is a second script from prev blocks
    
temp_root = r'C:\Users\Jacob Kim\Google Drive (jmhkim@stanford.edu)'
'''
temp_df1 = pd.read_csv(temp_root + r"\Whi5pr_mutants\220507_SM_yJK098_0ngatc_data_good_positions\output_df.csv")
temp_df2 = pd.read_csv(temp_root + r"\Whi5pr_mutants\220509_SM_yJK098_0ngatc_data_good_positions\output_df.csv")
#temp_df3 = pd.read_csv(r"G:\My Drive\Whi5pr_mutants\220408_yJK098-4_5ngmLatc-acdc\output_df.csv")
temp_df4 = pd.read_csv(temp_root + r"\Whi5pr_mutants\220503_JK_SM_yJK098-4_expt2repeat\good_positions\output_df.csv")
temp_df5 = pd.read_csv(temp_root + r"\Whi5pr_mutants\220506_SM_yJK098_5ngaTc_expt3\output_df.csv")
temp_df6 = pd.read_csv(temp_root + r"\ffc_test\bulktest\acdc_files\220607_SM_yJK098_30ngmLaTc_expt3_data\output_df.csv")
temp_df7 = pd.read_csv(temp_root + r"\Whi5pr_mutants\220523_yJK098_30ngmLaTc_expt1\output_df.csv")
overall_df_with_rel = pd.concat([temp_df1,temp_df2,temp_df4,temp_df5,temp_df6,temp_df7],ignore_index=True)
'''
temp_df1 = pd.read_csv(temp_root + r"\Whi5pr_mutants\220510_SM_yJK100_0ngatc_data_good_positions_expt1\output_df.csv")
temp_df2 = pd.read_csv(temp_root + r"\Whi5pr_mutants\220511_SM_yJK100_0ngatc_data_good_positions_expt2\output_df.csv")
temp_df3 = pd.read_csv(temp_root + r"\Whi5pr_mutants\220524_JK_yJK100_5ngmLaTc_expt3\output_df.csv")
temp_df4 = pd.read_csv(temp_root + r"\Whi5pr_mutants\220525_yJK100_5ngmLaTc_expt4\output_df.csv")
temp_df5 = pd.read_csv(temp_root + r"\ffc_test\bulktest\acdc_files\220603_yJK100_30ngmLaTc_expt1_data\output_df.csv")
#temp_df6 = pd.read_csv(temp_root + r"\ffc_test\bulktest\acdc_files\220605_SM_yJK100_30ngaTc_expt2_data\output_df.csv")
overall_df_with_rel = pd.concat([temp_df1,temp_df2,temp_df4,temp_df5],ignore_index=True)
#overall_df_with_rel2 = pd.concat([temp_df1,temp_df2,temp_df3,temp_df4,temp_df5],ignore_index=True)
#overall_df_with_rel3 = pd.concat([overall_df_with_rel,overall_df_with_rel2],ignore_index = True)
ch_name = 'mCitrine'


# In[28]:


#overall_df_with_rel['expt'] = 'yJK098-4 5ng_aTc expt2'


# # Plot testing

# In[234]:
### Functionalize this block

# some configurations
# frame interval of video
frame_interval_minutes = 6
# minimum number of cell cycles contributing to the mean+CI curve:
min_no_of_ccs = 0
# determine if you want to split the plot by generation
split_by_gen = False

# get indices of cells after filtering by criteria
#filter_idx = np.logical_and(
#    overall_df_with_rel['complete_cycle']==1, 
#    overall_df_with_rel.selection_subset==0, 
#    overall_df_with_rel.generation_num==1
#    overall_df_with_rel.cell_cycle_stage=='G1', )


# get indices of cells after filtering by criteria
filter_idx = (overall_df_with_rel['complete_cycle']==1) &     (overall_df_with_rel.generation_num==1) #&\
    #(overall_df_with_rel.cell_cycle_stage=='G1') 
#(overall_df_with_rel.selection_subset==0)&


# select needed cols from overall_df_with_rel
needed_cols = [
    'cell_unique_id', 'file', 'position', 'Cell_ID', 'generation_num', 'selection_subset', # identifing info
    'cell_cycle_stage', 'frame_i', 'phase_length', 'phase_begin', # time/age info
    'cell_vol_fl', 'cell_vol_fl_rel','cell_vol_vox', 'cell_vol_vox_rel', # size info
    f'{ch_name}_af_corrected_amount', f'{ch_name}_af_corrected_amount_rel', f'{ch_name}_af_corrected_concentration', f'{ch_name}_af_corrected_concentration_rel', # fluo info
    'relationship', 'relative_ID', 'mCitrine_sum', 'Strain', 'aTc', 'Replicate','Date' # relative info
]

# data to plot is a copy of the full dataframe, including only rows that meet the filter criterion and only taking columns needed for calculations/plotting
data = overall_df_with_rel.loc[filter_idx, needed_cols].copy()

# calculate the time the cell already spent in the current frame at the current timepoint
data['frames_in_phase'] = data['frame_i'] - data['phase_begin'] + 1
# calculate the time to the next (for G1 cells) and from the last (for S cells) G1/S transition  
data['bud_aligned_frames_in_phase'] = data.apply(
    lambda x: x.loc['frames_in_phase'] if x.loc['cell_cycle_stage']=='S' \
    else x.loc['frames_in_phase']-1-x.loc['phase_length'],
    axis=1
)
"""
data['birth_aligned_frames'] = data.apply(
    lambda x: x.loc['frames_in_phase']+x.loc[''] if x.loc['cell_cycle_stage']=='S' \
    else x.loc['frames_in_phase']-1-x.loc['phase_length'],
    axis=1
)
"""
# @todo move these variables to the first script
# TO DO: figure out how to get the birth-aligned frames

# calculate centered time in minutes
data['bud_aligned_time_in_minutes'] = data.bud_aligned_frames_in_phase * frame_interval_minutes

# calculate fluo signal contribution from bud
data['Bud amount'] = data.apply(
    lambda x: x.loc[f'{ch_name}_af_corrected_amount_rel'] if  x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' \
    else 0,
    axis=1
)
# combine mother/bud fluo signals
data['Combined m&b amount'] = data.apply(
    lambda x: x.loc[f'{ch_name}_af_corrected_amount'] + x.loc[f'{ch_name}_af_corrected_amount_rel'] if x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' \
    else x.loc[f'{ch_name}_af_corrected_amount'],
    axis=1
)
# combine mother/bud volumes
data['Combined m&b volume'] = data.apply(
    lambda x: x.loc['cell_vol_fl']+x.loc['cell_vol_fl_rel'] if\
    x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' else\
    x.loc['cell_vol_fl'],
    axis=1
)

# combine mother/bud volumes
data['Combined m&b volume vox'] = data.apply(
    lambda x: x.loc['cell_vol_vox']+x.loc['cell_vol_vox_rel'] if\
    x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' else\
    x.loc['cell_vol_vox'],
    axis=1
)
# final concentration is total Venus in mother/bud pair divided by their combined volume
data['Combined m&b concentration'] = data['Combined m&b amount']/data['Combined m&b volume']


# check that 'data' dataframe has necessary columns
expt_data = data

# TO DO: write something that saves the processed data for plotting into a csv
# later, have a separate script (functions?) for loading csvs from different movies, to plot together


# In[93]:


expt_data
#foo = expt_data.index.duplicated()
#len(foo)
#len(expt_data)


# In[236]:


# plot stuff

#x_var = 'frames_in_phase'
x_var = 'bud_aligned_time_in_minutes'
#y_var_to_plot = f'Venus_af_corrected_concentration'
y_var = 'Combined m&b amount'

# Generate figures (aggregated, single traces, combined
sns.set_theme(context='talk', font_scale=1.15)
sns.set_style("whitegrid", {"grid.color": ".95"})
#fig, axs = plt.subplots(ncols=3, figsize=(10,10), sharey=False)
sns.set(rc = {'figure.figsize':(10,10)},font_scale = 2,
        )
#sns.set_ylim(0,80000)
g1 = sns.lineplot(
    data=expt_data,
    x=x_var, 
    y=y_var,
    ci=95,
    #ax=axs[0],
    hue='aTc',
)
g1.set(ylim = (0,60000),xlabel = 'Time from Budding (min)', ylabel = 'Total Whi5-mCitrine (A.U.)')
#g1.set(ylim = (0,200),xlabel = 'Time from Budding (min)', ylabel = 'Volume (fL)')
'''
sns.lineplot(
    data=data,
    x=x_var, 
    y=y_var,
    estimator=None,
    units='cell_unique_id',
    ax=axs[1],
    lw=0.5,
    #alpha=0.5
)

sns.lineplot(
    data=data,
    x=x_var, 
    y=y_var,
    ci=95,
    ax=axs[2]
)

sns.lineplot(
    data=data,
    x=x_var, 
    y=y_var,
    estimator=None,
    units='cell_unique_id',
    ax=axs[2],
    lw=0.5,
    #alpha=0.5
)

axs[0].set_ylabel(f"Amount of {ch_name} (a.u.)")
#axs[2].set_xlabel(f"Time from budding (min)")
axs[1].set_ylabel(f"Amount of {ch_name} (a.u.)")
axs[2].set_ylabel(f"Amount of {ch_name} (a.u.)")
'''
"""
axs[0].set_xlabel(f"Volume (voxels)")
axs[2].set_xlabel(f"Time from birth (min)")
axs[1].set_ylabel(f"Volume (voxels)")
axs[2].set_ylabel(f"Volume (voxels)")
"""
#axs[0].set_ylim(0,0.00000006)
#axs[1].set_ylim(0,0.00000006)
#axs[2].set_ylim(0,50000000000)
#plt.savefig('../figures/firstgen_g1_concentration.png', dpi=300)
#plt.show()


# In[1]:


# plot old vs new camera

#x_var = 'frames_in_phase'
x_var = 'bud_aligned_time_in_minutes'
#y_var_to_plot = f'Venus_af_corrected_concentration'
y_var = 'Combined m&b amount'

# Generate figures (aggregated, single traces, combined
sns.set_theme(context='talk', font_scale=1.15)
sns.set_style("whitegrid", {"grid.color": ".95"})
fig, axs = plt.subplots(ncols=3, figsize=(30,10), sharey=False)

expt_data['adjusted_mCitrine'] = expt_data['Combined m&b amount']*12.386
#expt_data['adjusted_mCitrine'] = expt_data['Combined m&b volume vox'] * 5.8
sns.lineplot(
    data=expt_data[expt_data['camera']=='old'],
    x=x_var, 
    y='adjusted_mCitrine',
    ci=95,
    ax=axs[0]
)

sns.lineplot(
    data=expt_data[expt_data['camera']=='new'],
    x=x_var, 
    y=y_var,
    ci=95,
    ax=axs[0]
)

expt_data['adjusted_mCitrine'] = expt_data['mCitrine_af_corrected_amount']*12.386
#expt_data['adjusted_mCitrine'] = expt_data['Combined m&b volume vox'] * 5.8
sns.histplot(
    data=expt_data[expt_data['camera']=='old'],
    x='adjusted_mCitrine',
    kde=True,
    stat = 'density',
    ax=axs[1]
)
temp_data = expt_data[expt_data['camera']=='old']
print(np.nanmean(temp_data['mCitrine_af_corrected_amount']))
#axs[1].vlines(,color = 'b')
sns.histplot(
    data=expt_data[expt_data['camera']=='new'],
    x='mCitrine_af_corrected_amount',
    kde=True,
    color='orange',
    stat = 'density',
    ax=axs[1]
)
temp_data2 = expt_data[expt_data['camera']=='new']
print(np.nanmedian(temp_data2['mCitrine_af_corrected_amount'])/np.nanmedian(temp_data['mCitrine_af_corrected_amount']))
#axs[1].vlines(,color = 'r')
sns.lineplot(
    data=expt_data[expt_data['camera']=='old'],
    x=x_var, 
    y=y_var,
    ci=95,
    ax=axs[2]
)

sns.lineplot(
    data=expt_data[expt_data['camera']=='old'],
    x=x_var, 
    y=y_var,
    estimator=None,
    units='cell_unique_id',
    ax=axs[2],
    lw=0.5,
    #alpha=0.5
)

axs[0].set_ylabel(f"Amount of {ch_name} (a.u.)")
#axs[2].set_xlabel(f"Time from budding (min)")
axs[1].set_ylabel(f"Amount of {ch_name} (a.u.)")
axs[2].set_ylabel(f"Amount of {ch_name} (a.u.)")
"""
axs[0].set_xlabel(f"Volume (voxels)")
axs[2].set_xlabel(f"Time from birth (min)")
axs[1].set_ylabel(f"Volume (voxels)")
axs[2].set_ylabel(f"Volume (voxels)")
"""
#axs[0].set_ylim(0,0.00000006)
#axs[1].set_ylim(0,0.00000006)
#axs[2].set_ylim(0,50000000000)

#plt.savefig('../figures/firstgen_g1_concentration.png', dpi=300)
plt.show()

len(expt_data[expt_data['camera']=='old'])


# In[ ]:





# In[44]:


# plot old vs new camera

'220509_SM_yJK098_0ngatc_data_good_positions',
'220507_SM_yJK098_0ngatc_data_good_positions',
'220506_SM_yJK098_5ngaTc_expt3',
'good_positions'

#x_var = 'frames_in_phase'
x_var = 'bud_aligned_time_in_minutes'
#y_var_to_plot = f'Venus_af_corrected_concentration'
y_var = 'Combined m&b amount'

# Generate figures (aggregated, single traces, combined
sns.set_theme(context='talk', font_scale=1.15)
sns.set_style("whitegrid", {"grid.color": ".95"})
fig, axs = plt.subplots(ncols=3, figsize=(30,10), sharey=False)

expt_data['adjusted_mCitrine'] = expt_data['Combined m&b amount']*12.386
#expt_data['adjusted_mCitrine'] = expt_data['Combined m&b volume vox'] * 5.8
sns.lineplot(
    data=expt_data[expt_data['camera']=='old'],
    x=x_var, 
    y='adjusted_mCitrine',
    ci=95,
    ax=axs[0]
)

sns.lineplot(
    data=expt_data[expt_data['camera']=='new'],
    x=x_var, 
    y=y_var,
    ci=95,
    ax=axs[0]
)

expt_data['adjusted_mCitrine'] = expt_data['mCitrine_af_corrected_amount']*12.386
#expt_data['adjusted_mCitrine'] = expt_data['Combined m&b volume vox'] * 5.8
sns.histplot(
    data=expt_data[expt_data['camera']=='old'],
    x='adjusted_mCitrine',
    kde=True,
    stat = 'density',
    ax=axs[1]
)
temp_data = expt_data[expt_data['camera']=='old']
print(np.nanmean(temp_data['mCitrine_af_corrected_amount']))
#axs[1].vlines(,color = 'b')
sns.histplot(
    data=expt_data[expt_data['camera']=='new'],
    x='mCitrine_af_corrected_amount',
    kde=True,
    color='orange',
    stat = 'density',
    ax=axs[1]
)
temp_data2 = expt_data[expt_data['camera']=='new']
print(np.nanmedian(temp_data2['mCitrine_af_corrected_amount'])/np.nanmedian(temp_data['mCitrine_af_corrected_amount']))


axs[0].set_ylabel(f"Amount of {ch_name} (a.u.)")
#axs[2].set_xlabel(f"Time from budding (min)")
axs[1].set_ylabel(f"Amount of {ch_name} (a.u.)")
axs[2].set_ylabel(f"Amount of {ch_name} (a.u.)")
"""
axs[0].set_xlabel(f"Volume (voxels)")
axs[2].set_xlabel(f"Time from birth (min)")
axs[1].set_ylabel(f"Volume (voxels)")
axs[2].set_ylabel(f"Volume (voxels)")
"""
#axs[0].set_ylim(0,0.00000006)
#axs[1].set_ylim(0,0.00000006)
#axs[2].set_ylim(0,50000000000)

#plt.savefig('../figures/firstgen_g1_concentration.png', dpi=300)
plt.show()


# In[37]:


expt_data[expt_data['camera']=='new']


# In[237]:


"""
Calculating and plotting synthesis rates
"""

# some configurations
# frame interval of video
frame_interval_minutes = 6
# minimum number of cell cycles contributing to the mean+CI curve:
min_no_of_ccs = 0
# determine if you want to split the plot by generation
split_by_gen = False

# get indices of cells after filtering by criteria
#filter_idx = np.logical_and(
#    overall_df_with_rel['complete_cycle']==1, 
#    overall_df_with_rel.selection_subset==0, 
#    overall_df_with_rel.generation_num==1
#    overall_df_with_rel.cell_cycle_stage=='G1', )

overall_df_with_rel['expt'] = 'yJK098-4 5ng_aTc expt2'
# get indices of cells after filtering by criteria
filter_idx = (overall_df_with_rel['complete_phase']==1) &     (overall_df_with_rel.generation_num==1)&     (overall_df_with_rel.cell_cycle_stage=='S') 


# select needed cols from overall_df_with_rel
needed_cols = [
    'cell_unique_id', 'file', 'position', 'Cell_ID', 'generation_num', 'selection_subset', # identifing info
    'cell_cycle_stage', 'frame_i', 'phase_length', 'phase_begin', # time/age info
    'cell_vol_fl', 'cell_vol_fl_rel', # size info
    f'{ch_name}_af_corrected_amount', f'{ch_name}_af_corrected_amount_rel', f'{ch_name}_af_corrected_concentration', f'{ch_name}_af_corrected_concentration_rel', # fluo info
    'relationship', 'relative_ID','Strain', 'aTc', 'Replicate','Date'#'camera' # relative info
]

# data to plot is a copy of the full dataframe, including only rows that meet the filter criterion and only taking columns needed for calculations/plotting
data = overall_df_with_rel.loc[filter_idx, needed_cols].copy()

# calculate the time the cell already spent in the current frame at the current timepoint
data['frames_in_phase'] = data['frame_i'] - data['phase_begin'] + 1
# calculate the time to the next (for G1 cells) and from the last (for S cells) G1/S transition  
data['bud_aligned_frames_in_phase'] = data.apply(
    lambda x: x.loc['frames_in_phase'] if x.loc['cell_cycle_stage']=='S' \
    else x.loc['frames_in_phase']-1-x.loc['phase_length'],
    axis=1
)

# TO DO: figure out how to get the birth-aligned frames

# calculate centered time in minutes
data['bud_aligned_time_in_minutes'] = data.bud_aligned_frames_in_phase * frame_interval_minutes

# calculate fluo signal contribution from bud
data['Bud amount'] = data.apply(
    lambda x: x.loc[f'{ch_name}_af_corrected_amount_rel'] if  x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' \
    else 0,
    axis=1
)
# combine mother/bud fluo signals
data['Combined m&b amount'] = data.apply(
    lambda x: x.loc[f'{ch_name}_af_corrected_amount'] + x.loc[f'{ch_name}_af_corrected_amount_rel'] if x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' \
    else x.loc[f'{ch_name}_af_corrected_amount'],
    axis=1
)
# combine mother/bud volumes
data['Combined m&b volume'] = data.apply(
    lambda x: x.loc['cell_vol_fl']+x.loc['cell_vol_fl_rel'] if\
    x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' else\
    x.loc['cell_vol_fl'],
    axis=1
)
# final concentration is total Venus in mother/bud pair divided by their combined volume
data['Combined m&b concentration'] = data['Combined m&b amount']/data['Combined m&b volume']


# check that 'data' dataframe has necessary columns
data
data['adjusted_mCitrine'] = data['Combined m&b amount']*12.386

# TO DO: write something that saves the processed data for plotting into a csv
# later, have a separate script (functions?) for loading csvs from different movies, to plot together


# In[239]:


"""
Plot synthesis rates
"""

x_var = 'bud_aligned_time_in_minutes'
#y_var_to_plot = f'Venus_af_corrected_concentration'
y_var = 'Combined m&b amount'

normalize_plot = False

unique_cell_ids = list(set(data["cell_unique_id"]))
unique_cell_ids_num = len(unique_cell_ids)

# Generate figures (aggregated, single traces, combined
sns.set_theme(context='talk', font_scale=1.15)
sns.set_style("whitegrid", {"grid.color": ".95"})
#fig, axs = plt.subplots(ncols=4, nrows = 2, figsize=(40,20), sharey=False)

budding_vols = []
synth_rates = []
aTc = []
strain = []
filename = []
rsquare = []
synthrate_df = pd.DataFrame()
for n in range(unique_cell_ids_num):
    data_bools = data["cell_unique_id"] == unique_cell_ids[n]
    unique_df = data[data_bools]
    temp_x = np.array(unique_df[x_var]).reshape((-1,1))
    temp_y = np.array(unique_df[y_var])
    #type(temp_x)
    temp_regr = linear_model.LinearRegression()
    temp_regr.fit(temp_x, temp_y)
    temp_r_squared = temp_regr.score(temp_x,temp_y)
    if temp_r_squared > 0.75:
        rsquare.append(temp_r_squared)
        synth_rates.append(temp_regr.coef_[0])
        
        first_frame_bool = unique_df["frames_in_phase"] == 1
        first_frame_vols = unique_df[first_frame_bool]["Combined m&b volume"]
        budding_vols.append(first_frame_vols.values[0])
        first_frame_aTc = unique_df[first_frame_bool]["aTc"]
        aTc.append(first_frame_aTc.values[0])
        first_frame_strain = unique_df[first_frame_bool]["Strain"]
        strain.append(first_frame_strain.values[0])
        first_frame_filename = unique_df[first_frame_bool]["file"]
        filename.append(first_frame_filename.values[0])
    else:
        None
synthrate_df['budding_vols'] = budding_vols
synthrate_df['synth_rates'] = synth_rates
synthrate_df['aTc'] = aTc
synthrate_df['strain'] = strain
synthrate_df['filename'] = filename
synthrate_df['rsquare'] = rsquare
synthrate_df = synthrate_df.sort_values('budding_vols')
#fig, axs = plt.subplots(ncols=2, figsize=(20,10), sharey=True)


if normalize_plot == True:
    x_norm = np.mean(budding_vols)
    y_norm = np.mean(synth_rates)
else:
    x_norm = np.ones(1)
    y_norm = np.ones(1)
sns.set(rc = {'figure.figsize':(10,10)},font_scale = 2)

#g0= sns.regplot(x=budding_vols/x_norm, y=synth_rates/y_norm, x_bins= [0.75,1,1.25,1.5,1.75,2,2.25])
g0= sns.regplot(x=budding_vols/x_norm, y=synth_rates/y_norm,x_bins= [50,75,100,125,150])
g0.set(title = '545bp promoter combined')
if normalize_plot == False:
    g0.set(xlim = (0,250),ylim=(0,400),ylabel = "Whi5 Synthesis Rate", xlabel='Volume at Budding (fL)')
else:
    g0.set(xlim = (0,3),ylim=(0,3),ylabel = "Whi5 Synthesis Rate (normalized)", xlabel='Volume at Budding (normalized)')

Y = np.array(synth_rates/y_norm)
X = np.array(budding_vols/x_norm)
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print("n=" + str(len(X)))
print("slope:" + str(results.params[1]))
print("95% confidence intervals:" + str(results.conf_int(0.05)[1]))

'''
if normalize_plot == True:
    x_norm = np.mean(temp_budding_vols)
    y_norm = np.mean(temp_synth_rates)
else:
    x_norm = np.ones(1)
    y_norm = np.ones(1)
'''

sns.set(font_scale = 2)
g1= sns.lmplot(
    data = synthrate_df, x='budding_vols', y='synth_rates',hue = 'strain',height = 10,
    aspect = 1)#,x_bins = [50,70,90,110,130,150])
#sns.pointplot(data = synthrate_df,x='budding_vols', y='synth_rates',hue = 'aTc',)
#sns.set(font_scale = 1)
g1.set(xlim = (0,250),ylim=(0,400),ylabel = "Whi5 Synthesis Rate", xlabel='Volume at Budding (fL)')
#g1.xlabel('Volume at Budding (fL)')
#g1.ylabel('Whi5 Synthesis Rate')

'''
if normalize_plot == False:
    g1.set(xlim = (0,200),ylim=(0,300),ylabel = "Whi5 Synthesis Rate", xlabel='Volume at Budding (fL)')
else:
    g1.set(xlim = (0,2),ylim=(0,2),ylabel = "Whi5 Synthesis Rate", xlabel='Volume at Budding (voxel)')


Y = np.array(synth_rates/y_norm)
X = np.array(budding_vols/x_norm)
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print("n=" + str(len(X)))
print("slope:" + str(results.params[1]))
print("95% confidence intervals:" + str(results.conf_int(0.05)[1]))
'''

#plt.legend(loc='lower right')



a# In[ ]:
#sns.lmplot(data = synthrate_df, x='synth_rates', y='rsquare',hue = 'aTc',height = 10, aspect = 1)
sns.histplot(data = synthrate_df, x = 'rsquare',binwidth = 0.01)



# In[46]:


"""
Plot synthesis rates
"""

x_var = 'bud_aligned_time_in_minutes'
#y_var_to_plot = f'Venus_af_corrected_concentration'
y_var = 'Combined m&b amount'

normalize_plot = False

unique_cell_ids = list(set(data["cell_unique_id"]))
unique_cell_ids_num = len(unique_cell_ids)

# Generate figures (aggregated, single traces, combined
sns.set_theme(context='talk', font_scale=1.15)
sns.set_style("whitegrid", {"grid.color": ".95"})
#fig, axs = plt.subplots(ncols=4, nrows = 2, figsize=(40,20), sharey=False)

budding_vols = []
synth_rates = []
aTc = []
strain = []
filename = []
rsquare = []
synthrate_df = pd.DataFrame()
for n in range(unique_cell_ids_num):
    data_bools = data["cell_unique_id"] == unique_cell_ids[n]
    unique_df = data[data_bools]
    temp_x = np.array(unique_df[x_var]).reshape((-1,1))
    temp_y = np.array(unique_df[y_var])
    #type(temp_x)
    temp_regr = linear_model.LinearRegression()
    temp_regr.fit(temp_x, temp_y)
    temp_r_squared = temp_regr.score(temp_x,temp_y)
    rsquare.append(temp_r_squared)
    synth_rates.append(temp_regr.coef_[0])
    first_frame_bool = unique_df["frames_in_phase"] == 1
    first_frame_vols = unique_df[first_frame_bool]["Combined m&b volume"]
    budding_vols.append(first_frame_vols.values[0])
    first_frame_aTc = unique_df[first_frame_bool]["aTc"]
    aTc.append(first_frame_aTc.values[0])
    first_frame_strain = unique_df[first_frame_bool]["Strain"]
    strain.append(first_frame_strain.values[0])
    first_frame_filename = unique_df[first_frame_bool]["file"]
    filename.append(first_frame_file.values[0])

unique_expts = set(experiment)
expt_nums = len(unique_expts)
fig, axs = plt.subplots(ncols=2, figsize=(20,10), sharey=True)
    

if normalize_plot == True:
    x_norm = np.mean(budding_vols)
    y_norm = np.mean(synth_rates)
else:
    x_norm = np.ones(1)
    y_norm = np.ones(1)
sns.set(rc = {'figure.figsize':(10,10)})
g0= sns.regplot(x=budding_vols/x_norm, y=synth_rates/y_norm,ax=axs[0],)

if normalize_plot == False:
    g0.set(xlim = (0,200),ylim=(0,300),ylabel = "Whi5 Synthesis Rate", xlabel='Volume at Budding (fL)')
else:
    g0.set(xlim = (0,2),ylim=(0,2),ylabel = "Whi5 Synthesis Rate", xlabel='Volume at Budding (voxel)')

Y = np.array(synth_rates/y_norm)
X = np.array(budding_vols/x_norm)
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print("n=" + str(len(X)))
print("slope:" + str(results.params[1]))
print("95% confidence intervals:" + str(results.conf_int(0.05)[1]))


labels=['new','old']
for m in range(expt_nums):
    expt_bools = [x == list(unique_expts)[m] for x in experiment]
    temp_budding_vols = []
    temp_synth_rates = []
    for eb in range(len(expt_bools)):
        if expt_bools[eb]:
            
            temp_budding_vols.append(budding_vols[eb])
            if m == 1:
                temp_synth_rates.append(synth_rates[eb])
            else:
                None
        else:
            None
    if normalize_plot == True:
        x_norm = np.mean(temp_budding_vols)
        y_norm = np.mean(temp_synth_rates)
    else:
        x_norm = np.ones(1)
        y_norm = np.ones(1)

    g1= sns.regplot(x=temp_budding_vols/x_norm, y=temp_synth_rates/y_norm,ax=axs[1],label = labels[m])
    if normalize_plot == False:
        g1.set(xlim = (0,200),ylim=(0,300),ylabel = "Whi5 Synthesis Rate", xlabel='Volume at Budding (fL)')
    else:
        g1.set(xlim = (0,2),ylim=(0,2),ylabel = "Whi5 Synthesis Rate", xlabel='Volume at Budding (voxel)')
    
    Y = np.array(synth_rates/y_norm)
    X = np.array(budding_vols/x_norm)
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    print("n=" + str(len(X)))
    print("slope:" + str(results.params[1]))
    print("95% confidence intervals:" + str(results.conf_int(0.05)[1]))

    
plt.legend(loc='lower right')


# In[50]:


set(experiment)


# In[46]:


x_var = 'bud_aligned_time_in_minutes'
#y_var_to_plot = f'Venus_af_corrected_concentration'
y_var = 'Combined m&b amount'

normalize_plot = False
normalize_by_WT = True
robust = True

unique_cell_ids = list(set(data["cell_unique_id"]))
unique_cell_ids_num = len(unique_cell_ids)

# Generate figures (aggregated, single traces, combined
sns.set_theme(context='talk', font_scale=1.15)
sns.set_style("whitegrid", {"grid.color": ".95"})
#fig, axs = plt.subplots(ncols=4, nrows = 2, figsize=(40,20), sharey=True)
names = ['220510_SM_yJK100_0ngatc_data_good_positions', '220511_SM_yJK100_0ngatc_data_good_positions', '220524_JK_yJK100_5ngmLaTc_expt3', '220525_yJK100_5ngmLaTc_expt4']


budding_vols = []
synth_rates = []
experiment = []
condition = []
for n in range(unique_cell_ids_num):
    data_bools = data["cell_unique_id"] == unique_cell_ids[n]
    unique_df = data[data_bools]
    temp_x = np.array(unique_df[x_var]).reshape((-1,1))
    temp_y = np.array(unique_df[y_var])
    #type(temp_x)
    temp_regr = linear_model.LinearRegression()
    temp_regr.fit(temp_x, temp_y)
    synth_rates.append(temp_regr.coef_[0])
    first_frame_bool = unique_df["frames_in_phase"] == 1
    first_frame_vols = unique_df[first_frame_bool]["Combined m&b volume"]
    budding_vols.append(first_frame_vols.values[0])
    first_frame_experiment = unique_df[first_frame_bool]["file"]
    experiment.append(first_frame_experiment.values[0])
    if first_frame_experiment.values[0] == names[0] or first_frame_experiment.values[0] == names[1]:
        condition.append("0ng/mL aTc")
    else:
        condition.append("5ng/mL aTc")
    
unique_expts = set(experiment)
unique_expts = sorted(unique_expts)
expt_nums = len(unique_expts)

unique_conds = set(condition)
unique_conds = sorted(unique_conds)
cond_nums = len(unique_conds)
fig, axs = plt.subplots(ncols=3, figsize=(30,10), sharey=True)
    

if normalize_plot == True:
    if normalize_by_WT == True:
        x_norm = np.ones(1) * 63.0702582249925
        y_norm = np.ones(1) * 98.01242155741132
    else:
        x_norm = np.mean(budding_vols)
        y_norm = np.mean(synth_rates)
    x_lim = (0,4)
    y_lim = (0,4)
    x_label = 'Normalized Volume at Budding (A.U.)'
    y_label = 'Normalized Whi5 Synthesis Rate (A.U.)'
else:
    x_norm = np.ones(1)
    y_norm = np.ones(1)
    x_lim = (0,150)
    y_lim = (0,600)
    x_label = 'Volume at Budding (fL)'
    y_label = 'Whi5 Synthesis Rate (A.U.)'
sns.set(rc = {'figure.figsize':(10,10)})
g0= sns.regplot(x=budding_vols/x_norm, y=synth_rates/y_norm,ax=axs[0],)
g0.set(xlim = x_lim,ylim=y_lim,ylabel = y_label, xlabel=x_label)
#g0.set(xlim = x_lim,ylim=y_lim,ylabel = y_label, xlabel=x_label)
Y = np.array(synth_rates/y_norm)
X = np.array(budding_vols/x_norm)
X = sm.add_constant(X)
if robust == True:
    model = sm.RLM(Y,X)
else:  
    model = sm.OLS(Y,X)
results = model.fit()
textstr = '\n'.join((
    "n=" + str(len(X)),
    "slope:" + str(results.params[1]),
    "95% confidence intervals:" + str(results.conf_int(0.05)[1])))
props = dict(boxstyle='round', facecolor='white')
axs[0].text(0.05, 0.95, textstr, transform=axs[0].transAxes, fontsize=13,
        verticalalignment='top', bbox=props)

temp_measurements = []

labels=[ '0ng/mL aTc expt 1', '0ng/mL aTc expt 2', '5ng/mL aTc expt 3', '5ng/mL aTc expt 4',]
for m in range(expt_nums):
    expt_bools = [x == list(unique_expts)[m] for x in experiment]
    temp_budding_vols = []
    temp_synth_rates = []
    for eb in range(len(expt_bools)):
        if expt_bools[eb]:
            temp_budding_vols.append(budding_vols[eb])
            temp_synth_rates.append(synth_rates[eb])
        else:
            None
    if normalize_plot == True:
        if normalize_by_WT == True:
            x_norm = np.ones(1) * 63.0702582249925
            y_norm = np.ones(1) * 98.01242155741132
        else:
            x_norm = np.mean(budding_vols)
            y_norm = np.mean(synth_rates)
    else:
        x_norm = np.ones(1)
        y_norm = np.ones(1)

    g1= sns.regplot(x=temp_budding_vols/x_norm, y=temp_synth_rates/y_norm,ax=axs[1],label = labels[m])
    g1.set(xlim = x_lim,ylim=y_lim, ylabel = y_label, xlabel=x_label)
    #g1.set(xlim = x_lim,ylim=y_lim, ylabel = y_label, xlabel=x_label)    
    
    Y = np.array(temp_synth_rates/y_norm)
    X = np.array(temp_budding_vols/x_norm)
    X = sm.add_constant(X)
    if robust == True:
        model = sm.RLM(Y,X)
    else:  
        model = sm.OLS(Y,X)
    results = model.fit()
    temp_measurements.append(("n=" + str(len(X)),"slope:" + str(results.params[1]),"95% confidence intervals:" + str(results.conf_int(0.05)[1])))
axs[1].legend(loc='lower right')
textstr = ''
for m in range(expt_nums):
    textstr += '\n'.join(('',
        labels[m] + ' ' + temp_measurements[m][0] + ':',
        temp_measurements[m][1],
        temp_measurements[m][2]))
    props = dict(boxstyle='round', facecolor='white')
    axs[1].text(0.05, 0.95, textstr, transform=axs[1].transAxes, fontsize=13,
            verticalalignment='top', bbox=props)

labels=[ '0ng/mL aTc', '5ng/mL aTc']

temp_measurements = []
for m in range(cond_nums):
    cond_bools = [x == list(unique_conds)[m] for x in condition]
    temp_budding_vols = []
    temp_synth_rates = []
    for eb in range(len(cond_bools)):
        if cond_bools[eb]:
            temp_budding_vols.append(budding_vols[eb])
            temp_synth_rates.append(synth_rates[eb])
        else:
            None
    if normalize_plot == True:
        if normalize_by_WT == True:
            x_norm = np.ones(1) * 63.0702582249925
            y_norm = np.ones(1) * 98.01242155741132
        else:
            x_norm = np.mean(budding_vols)
            y_norm = np.mean(synth_rates)
    else:
        x_norm = np.ones(1)
        y_norm = np.ones(1)

    g2= sns.regplot(x=temp_budding_vols/x_norm, y=temp_synth_rates/y_norm,ax=axs[2],label = labels[m])
    g2.set(xlim = x_lim,ylim=y_lim, ylabel = y_label, xlabel=x_label)
    #g2.set(xlim = x_lim,ylim=y_lim, ylabel = y_label, xlabel=x_label)    
    Y = np.array(temp_synth_rates/y_norm)
    X = np.array(temp_budding_vols/x_norm)
    X = sm.add_constant(X)
    if robust == True:
        model = sm.RLM(Y,X)
    else:  
        model = sm.OLS(Y,X)
    results = model.fit()
    
    temp_measurements.append(("n=" + str(len(X)),"slope:" + str(results.params[1]),"95% confidence intervals:" + str(results.conf_int(0.05)[1])))
    
axs[2].legend(loc='lower right')

textstr = ''
for m in range(cond_nums):
    textstr += '\n'.join(('',
        list(unique_conds)[m] + ' ' + temp_measurements[m][0] + ':',
        temp_measurements[m][1],
        temp_measurements[m][2]))
    props = dict(boxstyle='round', facecolor='white')
    axs[2].text(0.05, 0.95, textstr, transform=axs[2].transAxes, fontsize=13,
            verticalalignment='top', bbox=props)


# In[36]:


print(sorted(unique_expts))


# In[34]:


if split_by_gen:
    g_cols = ['bud_aligned_frames_in_phase', 'Generation']
else:
    g_cols = 'bud_aligned_frames_in_phase'
    
# group dataframe to calculate sample sizes per generation
standard_grouped = data.groupby(  ['position', 'file', 'Cell_ID', 'generation_num']  ).agg('count').reset_index()
data['Generation'] = data.apply(
    lambda x: f'1st ($n_1$={len(standard_grouped[standard_grouped.generation_num==1])})' if\
    x.loc['generation_num']==1 else f'2+ ($n_2$={len(standard_grouped[standard_grouped.generation_num>1])})',
    axis=1
)
data['contributing_ccs_at_time'] = data.groupby(g_cols).transform('count')['selection_subset']
data = data[data.contributing_ccs_at_time >= min_no_of_ccs]

# finally prepare data for plot (use melt for multiple lines)
sample_size = len(standard_grouped)
avg_cell_cycle_length = round(standard_grouped.loc[:,'bud_aligned_time_in_minutes'].mean())*frame_interval_minutes
cols_to_plot = ['Combined m&b concentration']
index_cols = [col for col in data.columns if col not in cols_to_plot]

data_melted = pd.melt(
    data, index_cols, var_name='Method of calculation'
).sort_values('Method of calculation')
#data_dir = os.path.join('..', 'data', 'paper_plot_data')
# save preprocessed data
#data.to_csv(os.path.join(data_dir, 'data.csv'), index=False)
#data.to_csv(os.path.join(data_dir, 'data.csv'), index=False)

#data_melted

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

# In[20]:


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


# In[5]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)
regr.coeff_
# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)



plt.show()


# In[3]:


regr.coef_


# In[4]:


fit(diabetes_X_test, diabetes_y_test,weight=None)


# In[ ]:




