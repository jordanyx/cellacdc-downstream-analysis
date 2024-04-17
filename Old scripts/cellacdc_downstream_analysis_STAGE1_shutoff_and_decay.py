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

import seaborn as sns
sns.set_theme()
from cellacdc import cca_functions
from cellacdc import myutils

# import numpy as np
from scipy.stats import sem
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

import cellacdcAnalysisUtils as acdc_utils

# In[]: 0 MISC USER INPUTS

frame_interval_minutes = 6 # frame interval of timecourse

# fluo info and other 
ch_name = 'mCitrineFFC' # name of fluo channel of interest for autofluorescence corrections; must be same as whatever is in the experiment metadata
overall_filepath = 'G:\My Drive\JX_DATA\Whi5-mCitrine shutoff\Whi5-mCitrine shutoff all/'
# params updated 7/8/22, provided by JK
a = 5.2615323399390606e-06 # units of au/pix/vox
b = 4.544630239072624 # units of au/pix

# NEW PARAMS FROM MS380 NO DRUG 24 MIN
# a = 5.698559309198345e-06
# b = 4.544273850408382

df_force_recalc = True # whether to recalculate the dataframe in case anything upstream of this script changes

# In[]: 1 LOADING DATA, GUI OPTION

# only load ONE EXPERIMENT at a time! otherwise all data will save into the first loaded experiment's folder
data_dirs, positions = cca_functions.configuration_dialog()
file_names = [path.basename(data_dirs[0])] #you can grab the outermost folder/file in a path using path.basename
# file_names = [path.split(p)[-1] for p in data_dirs]
image_folders = [[path.join(data_dir, pos_str, 'Images') for pos_str in pos_list] for pos_list, data_dir in zip(positions, data_dirs)]
# determine available channels based on first(!) position.
# Warn user if one or more of the channels are not available for some positions
first_pos_dir = path.join(data_dirs[0], positions[0][0], 'Images')
first_pos_files = myutils.listdir(first_pos_dir)
channels, warn = cca_functions.find_available_channels(first_pos_files, first_pos_dir)

# =============================================================================
# # In[]: LOADING DATA, MANUAL OPTION
# 
# # only load ONE EXPERIMENT at a time! otherwise all data will save into the first loaded experiment's folder
# data_dirs =  [
#                 'C:/Users/jyxiao/DATA/220502_SM_JX61a', # WT clone 1 expt 1
#              ]
# 
# positions = [
#                 ['Position_1', 'Position_2', 'Position_3', 'Position_4', 'Position_5', 'Position_6', 'Position_7'],
#             ]
# 
# file_names = [path.basename(data_dirs[0])]
# image_folders = [[path.join(data_dir, pos_str, 'Images') for pos_str in pos_list] for pos_list, data_dir in zip(positions, data_dirs)]
# # determine available channels based on first(!) position.
# # Warn user if one or more of the channels are not available for some positions
# first_pos_dir = path.join(data_dirs[0], positions[0][0], 'Images')
# first_pos_files = myutils.listdir(first_pos_dir)
# channels, warn = cca_functions.find_available_channels(first_pos_files, first_pos_dir)
# 
# =============================================================================


# In[]: 2 GENERATE OVERALL DATAFRAME
# overall_df is essentially acdc_output.csv with additional columns calculated.
# this is already a function, can leave as is

overall_df, is_timelapse_data, is_zstack_data = cca_functions.calculate_downstream_data(
    file_names,
    image_folders,
    positions,
    channels, 
    force_recalculation = df_force_recalc # if True, recalculates overall_df; use this if anything upstream is changed
)
df_force_recalc = False


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

# In: GENERATE FIELDS USEFUL FOR BOOK-KEEPING
# keep this exposed and mutable; annotations may depend on user/experiment type
# NOTE: you can also add custom fields to the acdc_output.csv via the GUI

# calculate a unique cell id across files by appending file, position, cell id, generation number (more precisely, this is cell cycle number, since we include generation number)
overall_df_with_rel['cell_unique_id'] = overall_df_with_rel.apply(
    lambda x: f'{x["file"]}_{x["position"]}_Cell_{x["Cell_ID"]}_Gen_{int(x["generation_num"])}', axis=1)


# In: CALCULATE VARIOUS USEFUL QUANTITIES
# NOTE: some of these may become redundant if some calculations are moved to the ACDC GUI save step instead
# in particular, autofluo will probably be moved upstream

# phase timing quantities (phase lengths, time in phase)
overall_df_with_rel = acdc_utils.calculate_phase_timing_quantities(overall_df_with_rel.copy(), frame_interval_minutes)

# autofluorescence corrections (fluo amount, fluo concentration)
overall_df_with_rel = acdc_utils.calculate_autofluo_corrected_quantities(overall_df_with_rel.copy(), ch_name, a, b)

# combined mother/bud quantities (volume, fluo amount, fluo concentration)
overall_df_with_rel = acdc_utils.calculate_combined_mother_bud_quantities(overall_df_with_rel.copy(), ch_name)


# In: SAVE STAGE 1 OUTPUT

df_all = overall_df_with_rel.copy()

# save but make sure only one experiment is loaded
if len(file_names) == 1:
    expt_filepath = data_dirs[0]
    df_all.to_csv(path_or_buf = path.join(expt_filepath,file_names[0]+'_stage1_output_all.csv'), index=False) # save to experiment folder
    print('Data saved to experiment folder: ' + expt_filepath)
    df_all.to_csv(path_or_buf = path.join(overall_filepath,file_names[0]+'_stage1_output_all.csv'), index=False) # save to overall folder for all experiments of the same category
    print('Data also saved to overall folder: ' + overall_filepath)

'''
END OF STAGE 1 ANALYSIS; MOVE TO STAGE 2 FROM HERE.
'''

# In[]: HALF LIFE AND DECAY PLOTTING

# EXPERIMENT & FIT PARAMS
# =============================================================================
# decay_start_time = 360 # realtime (min) where tet is shut off
# frame_interval_minutes = 6 # frame interval of timecourse
# frame_end = 121 # last frame to fit
# frame_start = 80 # first frame to fit; PICK SOMETHING ABOUT TWO HOURS AFTER SHUTOFF
# 
# =============================================================================

decay_start_time = 360 # realtime (min) where tet is shut off
frame_interval_minutes = 6 # frame interval of timecourse
frame_end = 121 # last frame to fit
frame_start = 80 # first frame to fit; PICK SOMETHING ABOUT TWO HOURS AFTER SHUTOFF


# =============================================================================
# decay_start_time = 0 # realtime (min) where tet is shut off
# frame_interval_minutes = 24 # frame interval of timecourse
# frame_end = 31 # last frame to fit
# frame_start = 5 # first frame to fit; PICK SOMETHING ABOUT TWO HOURS AFTER SHUTOFF
# 
# =============================================================================

filter_idx = (df_all['is_cell_dead'] == 0)      \
            &(df_all['is_cell_excluded'] == 0)  \
            # &(df_all['generation_num'] != 0) # exclude buds, since they're counted 

needed_cols = ['file', 'position', 'Cell_ID', 'cell_unique_id', 'frame_i',
               'cell_area_pxl', 'cell_vol_fl', f'{ch_name}_af_corrected_amount', f'{ch_name}_af_corrected_concentration',
               'Combined m&b volume fL', 'Combined m&b amount', 'Combined m&b concentration']

df_filt = df_all.loc[filter_idx, needed_cols].copy()
df_filt['realtime_min'] = df_filt['frame_i'] * frame_interval_minutes # experiment time (min)
df_filt['relative_time'] = df_filt['realtime_min'] - decay_start_time # time (min) after t_shutoff = 0

# df_filt['fluo test'] = df_filt['Combined m&b amount'] - 30*df_filt['Combined m&b volume fL']

fluo_quant = f'{ch_name}_af_corrected_amount' # quantity to fit to
vol_quant = 'cell_vol_fl'
area_quant = 'cell_area_pxl'
# fluo_quant = 'Combined m&b amount'
# vol_quant = 'Combined m&b volume fL'

timepoints = np.array(range(frame_end)) * frame_interval_minutes - decay_start_time # time setting shutoff to 0

numcells = np.zeros(len(timepoints))
totalsig = np.zeros(len(timepoints))
meansig = np.zeros(len(timepoints))
fluostderror = np.zeros(len(timepoints))

totalvol = np.zeros(len(timepoints))
meanvol = np.zeros(len(timepoints))
volstderror = np.zeros(len(timepoints))

totalarea = np.zeros(len(timepoints))
meanarea = np.zeros(len(timepoints))
areastderror = np.zeros(len(timepoints))

for t in range(len(timepoints)):
    temp_df = df_filt.loc[df_filt['relative_time'] == timepoints[t]].copy() # get all rows that correspond to the current timepoint
    numcells[t] = temp_df[fluo_quant].size
    
    totalsig[t] = np.nansum(temp_df[fluo_quant]) # total fluo signal of all cells in the current timepoint
    meansig[t] = np.nanmean(temp_df[fluo_quant]) # total fluo signal divided by number of cells
    fluostderror[t] = sem(temp_df[fluo_quant], nan_policy='omit') # standard error of the mean of total fluo signal
    
    totalvol[t] = np.nansum(temp_df[vol_quant]) # total volume of all cells in the current timepoint
    meanvol[t] = np.nanmean(temp_df[vol_quant])
    volstderror[t] = sem(temp_df[vol_quant], nan_policy='omit')
    
    totalarea[t] = np.nansum(temp_df[area_quant]) # total volume of all cells in the current timepoint
    meanarea[t] = np.nanmean(temp_df[area_quant])
    areastderror[t] = sem(temp_df[area_quant], nan_policy='omit')

totalconc = totalsig/totalvol # total fluo signal divided by total cellular volume


# to do: total "concentration": total mCitrine divided by total cell volume
# compare this to perfect dilution, which is initial mCitrine divided by total cell volume at each point in time

# In[]
# histogram of autofluo mean mCitrineFFC in each cell vs. this movie's, last frame for each

vals = df_filt.loc[df_filt['frame_i']==30][f'{ch_name}_af_corrected_amount']

g0 = sns.histplot(vals)
plt.axvline(np.nanmean(vals))
plt.axvline(np.nanmedian(vals))
plt.title(file_names[0] + ', mean = ' + str(np.nanmean(vals)))
# g0.set(xlim = (0,15000))

# In[]: LINEAR AND EXPO FIT BY TOTAL SIGNAL, i.e. total fluo at each timepoint

xvals = timepoints[frame_start:frame_end]
yvals_raw = totalsig[frame_start:frame_end] # / totalvol[frame_start:frame_end]
norm_factor = totalsig[frame_start] # change to 1 to remove norm
# norm_factor = 1
yvals = yvals_raw/norm_factor # normalize to starting point from which decay is calculated

# get linfit separately because regplot is dumb and doesn't return it
m , b = np.polyfit(xvals,yvals,1)
y_pred = m * xvals + b

# expo portion
# initial parameter guesses
a = yvals[0] # starting height
b = (yvals[-1]-yvals[0]) / (xvals[-1]-xvals[0]) * 1/np.mean(yvals) # approximate linear slope
c = 0 # constant offset

# use predefined exponential function with the curve_fit function from scipy
# params, covs = curve_fit(f = acdc_utils.expo_decay_fit, xdata=xvals, ydata=yvals, p0 = [a,b,c])
# half_life = np.log( (yvals[0]*0.5 - params[2]) / params[0] ) / params[1]
# print(params) # in the same order as the parameter guesses; middle value is the decay constant
# print('Half life: ' + str(half_life) + 'minutes')


sns.set(font_scale=1.5)

FIG1, ax1 = plt.subplots()
FIG1.set_size_inches(8, 6)

FIG1 = sns.regplot(x=xvals,y=yvals) # linear fit
plt.plot(timepoints, totalsig/norm_factor) # actual data
# plt.plot(xvals,acdc_utils.expo_decay_fit(xvals,params[0],params[1],params[2])) # expo fit

ylim = [0,None]
#FIG1.set(xlim=[0,timepoints[-1]])
#FIG1.set(xlim = [240,360])
# FIG1.set(ylim=ylim)
FIG1.set_ylabel('Total fluo (all cells, normalized)')
FIG1.set_xlabel('Time relative to shutoff (min)')
FIG1.set_title(file_names[0] + ', slope = ' + str(m))

FIG1.legend(['Data fitted', 'linear fit', 'linfit 95% confidence', 'Full data', 'Expo fit'])


# =============================================================================
# # In[] confirming polyfit is same as regplot
# FIG = plt.subplot()
# plt.scatter(xvals,yvals)
# plt.plot(xvals,y_pred)
# FIG.set(ylim=ylim)
# FIG.set_ylabel('Total Whi5-mCit (normalized)')
# FIG.set_xlabel('Time after frame ' + str(frame_start) + ' (min)')
# =============================================================================


# In[] FIT BY MEAN SIGNAL, i.e. total fluo at each timepoint divided by number of cells; exponential fit

xvals = timepoints[frame_start:frame_end]
yvals = meansig[frame_start:frame_end] # MEAN PER CELL
yerr = fluostderror[frame_start:frame_end]
perfect_dilution = totalsig[frame_start] / numcells[frame_start:frame_end]

# initial parameter guesses
a = yvals[0] # starting height
b = (yvals[-1]-yvals[0]) / (xvals[-1]-xvals[0]) * 1/np.mean(yvals) # approximate linear slope
c = 0 # constant offset

# use predefined exponential function with the curve_fit function from scipy
params, covs = curve_fit(f = acdc_utils.expo_decay_fit, xdata=xvals, ydata=yvals, p0 = [a,b,c])

print(params) # in the same order as the parameter guesses; middle value is the decay constant

FIG3 = plt.errorbar(xvals,yvals,yerr,label='mean and std error')

plt.plot(xvals,acdc_utils.expo_decay_fit(xvals,params[0],params[1],params[2]),label='fit')
plt.plot(xvals, perfect_dilution,label='perfect dilution')
# plt.ylim(ylim)
plt.ylabel('Mean Whi5-mCit per cell')
plt.xlabel('Time relative to shutoff (min)')
plt.title(file_names[0])
plt.legend()

# In[] MEAN VOLUME, i.e. AVERAGE VOLUME PER CELL

xvals = timepoints[frame_start:frame_end]
yvals = meanvol[frame_start:frame_end] # MEAN PER CELL
yerr = volstderror[frame_start:frame_end]

FIG4 = plt.errorbar(xvals,yvals,yerr,label='mean and std error')

# plt.plot(xvals, perfect_dilution,label='perfect dilution')
# plt.ylim(ylim)
plt.ylabel('Mean volume (fL) per cell')
plt.xlabel('Time relative to shutoff (min)')
plt.title(file_names[0])
plt.legend()


# In[] MEAN AREA, i.e. AVERAGE AREA PER CELL

xvals = timepoints[frame_start:frame_end]
yvals = meanarea[frame_start:frame_end] # MEAN PER CELL
yerr = areastderror[frame_start:frame_end]

FIG4 = plt.errorbar(xvals,yvals,yerr,label='mean and std error')

# plt.plot(xvals, perfect_dilution,label='perfect dilution')
# plt.ylim(ylim)
plt.ylabel('Mean area (pixels) per cell')
plt.xlabel('Time relative to shutoff (min)')
plt.title(file_names[0])
plt.legend()


# In[] TOTAL CONCENTRATION, i.e. total fluo at each timepoint divided by total cellular volume; exponential fit

xvals = timepoints[frame_start:frame_end]
yvals = totalconc[frame_start:frame_end]
perfect_dilution = totalsig[frame_start] / totalvol[frame_start:frame_end]

# initial parameter guesses
a = yvals[0] # starting height
b = (yvals[-1]-yvals[0]) / (xvals[-1]-xvals[0]) * 1/np.mean(yvals) # approximate linear slope
c = 0 # constant offset

# use predefined exponential function with the curve_fit function from scipy
params, covs = curve_fit(f = acdc_utils.expo_decay_fit, xdata=xvals, ydata=yvals, p0 = [a,b,c])

print(params) # in the same order as the parameter guesses; middle value is the decay constant

FIG5 = plt.plot(xvals,yvals,label='data')

plt.plot(xvals,acdc_utils.expo_decay_fit(xvals,params[0],params[1],params[2]),label='fit')
plt.plot(xvals, perfect_dilution,label='perfect dilution')
plt.ylim(ylim)
plt.ylabel('Total concentration')
plt.xlabel('Time relative to shutoff (min)')
plt.title(file_names[0])
plt.legend()



# In[]
# FIG5, ax5 = plt.subplots()
# sns.relplot(data = df_filt, x = 'relative_time', y = quant_to_fit, hue = 'file')

# In[] PLOT INDIVIDUAL CELL TRACES
filter_idx = (df_filt['Cell_ID'] >= 5)    \
            &(df_filt['Cell_ID'] <= 5)    \
            &(df_filt['position'] == 'Position_1')  \
                
df_filt_more = df_filt.loc[filter_idx].copy()
FIG3, ax3 = plt.subplots()
x_var = 'relative_time'
y_var = fluo_quant

sns.set(font_scale=0.5)
sns.lmplot(data = df_filt_more, x = x_var, y = y_var, col = 'Cell_ID', col_wrap = 5)

# =============================================================================
# # In[] EXPO FIT TO TOTAL SIGNAL (NOW INTEGRATED WITH EARLIER PLOTTING BLOCK)
# xvals = timepoints[frame_start:frame_end]
# yvals = totalsig[frame_start:frame_end]
# norm_factor = totalsig[frame_start] # change to 1 to remove norm
# yvals = yvals/norm_factor # normalize to starting point from which decay is calculated
# 
# # initial parameter guesses
# a = yvals[0] # starting height
# b = (yvals[-1]-yvals[0]) / (xvals[-1]-xvals[0]) * 1/np.mean(yvals) # approximate linear slope
# c = 0 # constant offset
# 
# # use predefined exponential function with the curve_fit function from scipy
# params, covs = curve_fit(f = acdc_utils.expo_decay_fit, xdata=xvals, ydata=yvals, p0 = [a,b,c])
# FIG2, ax2 = plt.subplots()
# FIG2.set_size_inches(8, 6)
# 
# plt.plot(xvals,acdc_utils.expo_decay_fit(xvals,params[0],params[1],params[2]))
# plt.plot(timepoints,totalsig/norm_factor)
# ylim = [0,None]
# # plt.xlim([0,timepoints[-1]])
# plt.ylim(ylim)
# plt.ylabel('Total Whi5-mCit (normalized)')
# plt.xlabel('Time relative to shutoff (min)')
# plt.title('Slope = ' + str(m))
# 
# half_life = np.log( (yvals[0]*0.5 - params[2]) / params[0] ) / params[1]
# 
# print(params) # in the same order as the parameter guesses; middle value is the decay constant
# print('Half life: ' + str(half_life) + 'minutes')
# =============================================================================



# In[]: SANITY CHECK PLOTTING (OUTDATED FOR THIS FILE)
# use to make sure single cell traces for a given position look right, or that positions within experiment are similar
# keep this exposed and simple to use

# USER INPUT: filter for cells of interest
filter_idx = (df_all['is_cell_dead']==0)         \
            # &(df_all['generation_num']==1)         \
            # &(df_all['position'] == 'Position_6')  \
            # &(df_all['mutant'] == '12A')           \

df_filt = df_all.loc[filter_idx].copy()

# choose fields to plot
x_var = 'frame_i'
y_var = 'Combined m&b amount'
xlim = [0,120]
ylim = [0,None]

# overlaying by mutant, position, etc.
hue = 'position'
#hue = 'cell_unique_id'

# the line that does the actual plotting
FIG = sns.relplot(data=df_filt, x=x_var, y=y_var, kind='line', hue=hue, height=8, aspect=1.2, legend=False)

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
FIG.fig.suptitle('PtetO.7-Whi5-mCitrine, shutoff at 4 hrs (40 frames), total cells = ' + str(totcells))
FIG.axes[0,0].set_xlabel(x_var)
FIG.axes[0,0].set_ylabel(y_var)



