#!/usr/bin/env python
# coding: utf-8

# In[]: IMPORTS

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
plt.rcParams.update({'font.size': 22})
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
sns.set_theme()
from cellacdc import cca_functions
from cellacdc import myutils

from sklearn.linear_model import LinearRegression, HuberRegressor
from scipy.stats import gaussian_kde, sem
import statsmodels.api as sm    
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import pickle

# # configurations
# - follow the file selection dialog:
#     - select microscopy folder in first step
#     - select positions of the selected folder in second step
# - repeat to add more positions to the analysis
# - positions selected within one iteration of the dialog will be pooled together in the following analyses

# In[]: EXPERIMENT SELECTION

data_dirs, positions = cca_functions.configuration_dialog()
file_names = [os.path.split(path)[-1] for path in data_dirs]
image_folders = [[os.path.join(data_dir, pos_str, 'Images') for pos_str in pos_list] for pos_list, data_dir in zip(positions, data_dirs)]
# determine available channels based on first(!) position.
# Warn user if one or more of the channels are not available for some positions
first_pos_dir = os.path.join(data_dirs[0], positions[0][0], 'Images')
first_pos_files = myutils.listdir(first_pos_dir)
channels, warn = cca_functions.find_available_channels(first_pos_files, first_pos_dir)

data_dirs, positions, file_names

# # load data and perform all needed calculations on image data
# This script is to calculate autofluorescence from a movie with no signal in the Venus channel.

# In[]:

overall_df, is_timelapse_data, is_zstack_data = cca_functions.calculate_downstream_data(
    file_names,
    image_folders,
    positions,
    channels, 
    force_recalculation=False # if True, re-runs calculations; otherwise just loads from before
)


# In[]:
# =============================================================================
# 
# csvfile1 = "G:/My Drive/JX_DATA/220430_yJK056_autof.csv"
# overall_df = pd.read_csv(csvfile1)
# 
# =============================================================================
# csvfile2 = "G:/My Drive/JX_DATA/220415_yJK056_autof.csv"
# temp_df = pd.read_csv(csvfile2)
# overall_df = pd.concat([overall_df,temp_df],ignore_index=True)

# =============================================================================
# # In[]: LOWESS FIT (FROM JK); total volume vs. total FFC fluo per cell
# 
# current_df = overall_df.copy()
# 
# X = current_df['cell_vol_vox']
# Y = current_df['mCitrineFFC_sum']
# 
# g0= sns.scatterplot(
#     x=X,
#     y=Y,
#     #x_bins = 20
#     )
# 
# model3 = sm.nonparametric.lowess(Y,X,it = 3)
# model3 = np.insert(model3,0,0,axis = 0)
# 
# slope = (model3[-1][1] - model3[-2][1])/ (model3[-1][0] - model3[-2][0])
# y = slope * (500000 - model3[-1][0]) + model3[-1][1]
# 
# model3 = np.insert(model3,0,[500000,y],axis = 0) 
# model3 = np.sort(model3,axis=0)
# 
# lowess_func3 = interp1d(model3[:,0],model3[:,1])
# 
# xvals = np.linspace(0, max(X), 300)
# 
# g1 = sns.lineplot(xvals, lowess_func3(xvals),color = 'r')
# pickfile = open(r'C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\autof_lowess_func_MS380.pickle','wb')
# pickle.dump(lowess_func3,pickfile)
# pickfile.close()
# =============================================================================

# In[]:
    
# PARAMS
gate = 0000 # VOXELS; only if there's some reason to discard small cells
numbins = 101 # bins for getting mean of residuals later

# get data from overall dataframe; these variables are pandas Series
AF_cell_area = overall_df[f'cell_area_pxl'] # PIXELS
AF_cell_vol = overall_df[f'cell_vol_vox'] # VOXELS

# CHOOSE FLUO VALUE (e.g. different channel, with or without background subtraction)
ch_name = 'FFCsub' # Jacob named mCitrineRaw as just Raw in the file '220430_yJK056_autof.csv'
# NOTE within ACDC, there are multiple sum/total/amount fields within ACDC
# 'raw' and 'raw_sum' both refer to the total fluo within cell area, no corrections (they're redundant)
# 'corrected_amount' refers to after ACDC's background correction (aka their version of flatfield corr); by default it's autoBkgr

# AF_cell_fluo = overall_df[f'{ch_name}_corrected_amount'] # AU; CHANGE THIS LINE TO CHANGE WHICH CHANNEL TO FIT
AF_cell_fluo = overall_df[f'{ch_name}_sum'] #- AF_cell_area * overall_df[f'{ch_name}_autoBkgr_bkgrVal_median'] # AU; 
# AF_cell_fluo = overall_df[f'{ch_name}_sum'] - 77*AF_cell_area# AU; 
AF_fluoperpixel = AF_cell_fluo / AF_cell_area # AU/PIXEL

# gate cells by volume, to exclude buds and small out-of-focus cells
gated_cell_indices = np.where(overall_df[f'cell_vol_vox']>=gate)
gated_areas = AF_cell_area[gated_cell_indices[0]]
gated_vols = AF_cell_vol[gated_cell_indices[0]]
gated_fluo = AF_cell_fluo[gated_cell_indices[0]]
gated_fluoperpixel = AF_fluoperpixel[gated_cell_indices[0]]

# NOTE: treat every cell in AF dataset as its own object (even though it's technically timecourse data)

mask = ~np.isnan(gated_fluo) & ~np.isnan(gated_fluoperpixel)
gated_areas = gated_areas[mask]
gated_vols = gated_vols[mask]
gated_fluo = gated_fluo[mask]
gated_fluoperpixel = gated_fluoperpixel[mask]

xlim = [0,100000]
ylim = [0,20000]
residlim = [-3000,3000]

# In[7]: FIT 1: cell volume vs. total AF per cell

# variables for the fit: y = mx + b
x = np.array(gated_vols)
y = np.array(gated_fluo)
#y = np.array(gated_fluoperpixel)

# note to self: regression function requires 2D arrays, hence cast to np.array and then reshape
#fit = LinearRegression().fit(x.reshape(-1,1),y.reshape(-1,1)) # regular linear least squares regression
fit = HuberRegressor().fit(x.reshape(-1,1),y.reshape(-1,1)) # robust linear regression that reduces influence of outliers
m = fit.coef_[0]
b = fit.intercept_
#y_pred = m*x + b # equivalent to fit.predict for simple Linear Regression
y_pred = fit.predict(x.reshape(-1,1))

# calculate point density for scatter plots
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# calculate residuals and mean for residuals
resid = y - y_pred

# calculate mean curve for residuals based on binning
xy = np.vstack([x,resid])
sorted_xy = np.array(sorted(xy.T,key=lambda x: x[0])) # sorts xy tuples by ascending x value
bins = np.linspace(0,xlim[1],num=numbins) # generates equally spaced bins from 0 to 25000
bin_assignments = np.digitize(sorted_xy[:,0],bins) # for each tuple in sorted_xy, assigns a bin number based on x value
bin_means = [ np.nanmean(sorted_xy[bin_assignments == i][:,1]) for i in range(1, len(bins)) ] # gets the y-value mean for the tuples in each bin i
bin_sem = [ sem(sorted_xy[bin_assignments == i][:,1],nan_policy='omit') for i in range(1, len(bins)) ] # gets the y-value standard error of the mean for the tuples in each bin i

print('y = mx + b: m =',m,'b =', b)

# PLOTS FOR FIT AND RESIDUALS
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(25, 10))

# fit
ax1.scatter(x, y, c=z, s=10)
ax1.plot(x,y_pred,'b',linewidth=6)
# ax1.set_title(str(file_names) + ", FIT 1: cell vol vs. total AF")
ax1.set_xlabel('Cell volume (voxels), gating = '+str(gate))
ax1.set_ylabel('Total fluo (au)')
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.legend(['Data','Fit'])

# residuals
ax2.scatter(x, resid, c=z, s=10)
ax2.plot(x,x*0,'b',linewidth=6)
ax2.errorbar(bins[1:],bin_means,bin_sem,color='cyan',linewidth=3)
# ax2.set_title(str(file_names) + ", FIT 1 residuals")
ax2.set_xlabel('Cell volume (voxels), gating = '+str(gate))
ax2.set_ylabel('Actual minus predicted total fluo (au)')
ax2.set_xlim(xlim)
ax2.set_ylim(residlim)
ax2.legend(['Actual minus predicted','Zero line','Binned mean and SEM'])

# TO DO: write a block to save the fit parameters


# In[8]: FIT 2: cell volume vs. AF per pixel

# variables for the fit: y = mx + b
x = np.array(gated_vols)
y = np.array(gated_fluoperpixel)

# note to self: regression function requires 2D arrays, hence cast to np.array and then reshape
fit = LinearRegression().fit(x.reshape(-1,1),y) # regular linear least squares regression
#fit = HuberRegressor().fit(x.reshape(-1,1),y.reshape(-1,1)) # robust linear regression that reduces influence of outliers
m = fit.coef_[0]
b = fit.intercept_
#y_pred = m*x + b # equivalent to fit.predict for simple Linear Regression
y_pred = fit.predict(x.reshape(-1,1))

# calculate point density for scatter plots
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# calculate residuals and mean for residuals
resid = gated_fluo - y_pred*gated_areas

# calculate mean curve for residuals based on binning
xy = np.vstack([x,resid])
sorted_xy = np.array(sorted(xy.T,key=lambda x: x[0])) # sorts xy tuples by ascending x value
bins = np.linspace(0,xlim[1],num=numbins) # generates equally spaced bins from 0 to 25000
bin_assignments = np.digitize(sorted_xy[:,0],bins) # for each tuple in sorted_xy, assigns a bin number based on x value
bin_means = [ np.nanmean(sorted_xy[bin_assignments == i][:,1]) for i in range(1, len(bins)) ] # gets the y-value mean for the tuples in each bin i
bin_sem = [ sem(sorted_xy[bin_assignments == i][:,1],nan_policy='omit') for i in range(1, len(bins)) ] # gets the y-value standard error of the mean for the tuples in each bin i

print('y = mx + b: m =',m,'b =', b)

# PLOTS FOR FIT AND RESIDUALS
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(25, 10))

# fit
ax1.scatter(x,gated_fluo, c=z, s=10)
ax1.scatter(x,y_pred*gated_areas)
# ax1.scatter(x,y, c=z, s=10)
# ax1.scatter(x,y_pred)
# ax1.set_title(str(file_names) + ", FIT 2: cell vol vs. mean AF per pixel")
ax1.set_xlabel('Cell volume (voxels), gating = '+str(gate))
ax1.set_ylabel('Total fluo (au)')
ax1.set_xlim(xlim)
# ax1.set_ylim(ylim)
ax1.legend(['Data','Fit'])

# residuals
ax2.scatter(x,resid, c=z, s=10)
ax2.plot(x,x*0,'b',linewidth=6)
ax2.errorbar(bins[1:],bin_means,bin_sem,color='cyan',linewidth=3)
# ax2.set_title(str(file_names) + ", FIT 2 residuals")
ax2.set_xlabel('Cell volume (voxels), gating = '+str(gate))
ax2.set_ylabel('Actual minus predicted total fluo (au)')
ax2.set_ylabel('Actual minus predicted total fluo/pixel (au/pix)')
ax2.set_xlim(xlim)
# ax2.set_ylim(residlim)
ax2.legend(['Actual minus predicted','Zero line','Binned mean and SEM'])


# TO DO: write a block to save the fit parameters

# In[]: OTHER MISC PLOTTING

resid = gated_fluoperpixel - y_pred

plt.hist(resid, bins = np.arange(-3,3,0.1), density = True)
plt.plot([0,0],[0,1])
plt.xlim([-3,3])
plt.xlabel('Residuals (au/pix)')
plt.ylabel('Count')
plt.title('Gating = '+str(gate))
# sns.regplot(x = gated_vols, 
#             y = gated_fluo, 
#             #x_bins = [10000* x for x in range(10)],
#             )



