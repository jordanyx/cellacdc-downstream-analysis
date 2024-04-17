# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:36:50 2024

@author: jyxiao

ACDC OUTPUT ANALYSIS
STAGE 3: PLOTTING FOR STARVATION WHI5 REIMPORT EXPERIMENTS

Input: multiple CSVs from Stage 1
Output: plots!

"""

# In[]: IMPORT STATEMENTS
# keep exposed

from os import path
import sys
sys.path.append(r"C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis")
from glob import glob
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 30)
pd.set_option("display.max_colwidth", 150)

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()
#from cellacdc import cca_functions
#from cellacdc import myutils
from scipy.stats import linregress
from scipy.stats import ttest_ind
import cellacdcAnalysisUtils as acdc_utils

# this function prints values on top of a seaborn bar plot
def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01) + float(space)
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.2f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)

# In[]: MISC USER INPUTS

# WHICH EXPERIMENT SET TO LOOK AT

overall_filepath = 'G:\My Drive\JX_DATA\Whi5-mCitrine reimport\Whi5-mCitrine reimport all/'

# In[]: LOAD STAGE 1 OUTPUT

df_paths = glob(path.join(overall_filepath,'*.csv')) # list of all CSV files in the directory specified by overall_filepath

first_file = True
for csvfile in df_paths:
    if first_file == True:
        df_all = pd.read_csv(csvfile) # initialize dataframe on the first CSV loaded
        first_file = False
    else:
        temp_df = pd.read_csv(csvfile)
        df_all = pd.concat([df_all,temp_df],ignore_index=True) # after first CSV is loaded, everything else is appended to the same dataframe


# In[]: PRE-PLOTTING

frame_interval_minutes = 6
media_switch_frame = 61
movie_end_frame = 161

# USE THIS ARRAY TO CHANGE WHICH MUTANTS GET PLOTTED
mutants_to_plot = [
                    'WT',    
                    '5S',    
                    ]
df_all['to_plot'] = df_all.apply(
    lambda x: True if x.loc['mutant'] in mutants_to_plot \
    else False,
    axis=1
)

# mapping colors and patterns to each mutant
full_pal = sns.color_palette('colorblind')
color_mapping = {
    "WT":    full_pal[2], # green
    "5S":    full_pal[4], # light purple
    }

sub_pal = [color_mapping[item] for item in mutants_to_plot]

    
# USER INPUT: filter for cells of interest
filter_idx = (df_all['to_plot']==True)         \
            &(df_all['is_cell_excluded']==0)         \
            &(df_all['is_cell_dead']==0)         \
            &(df_all['complete_cycle']==1)         \
            &(df_all['Whi5_exit_frame']<=media_switch_frame)         \
            &(df_all['division_frame']>=media_switch_frame)         \
            # &(df_all['budding_frame']<=media_switch_frame)         \
            # &(df_all['division_frame']>=media_switch_frame)         \

df_filt = df_all.loc[filter_idx].copy()
df_filt.mutant = pd.Categorical(df_filt.mutant, categories=mutants_to_plot) # ensures mutants are plotted in consistent order


# In[]: FINAL CALCULATIONS

# time from Whi5 exit to media switch
df_filt['Whi5 exit to media switch frames'] = media_switch_frame - df_filt['Whi5_exit_frame']
df_filt['Whi5 exit to media switch minutes'] = df_filt['Whi5 exit to media switch frames'] * frame_interval_minutes

# time from budding to media switch
df_filt['Budding to media switch frames'] = media_switch_frame - df_filt['budding_frame']
df_filt['Budding to media switch minutes'] = df_filt['Budding to media switch frames'] * frame_interval_minutes

# time from media switch to reentry
df_filt['Media switch to Whi5 reentry frames'] = df_filt['Whi5_reentry_frame'] - media_switch_frame
df_filt['Media switch to Whi5 reentry minutes'] = df_filt['Media switch to Whi5 reentry frames'] * frame_interval_minutes


# In[]: TIME-DEPENDENT PLOTTING

# choose fields to plot
# x_var = 'minutes_in_phase'
x_var = 'time_i'

# y_var = 'mCitrineScaledFFC_CV'
y_var = 'Combined m&b volume fL'


xlim = [0,movie_end_frame*frame_interval_minutes]
ylim = [0, None] # everything else

# overlaying by mutant, position, etc.
# hue = 'file'
hue = 'mutant'
# hue = 'position'
# hue = 'cell_unique_id'

sns.set_theme(style='whitegrid', font_scale = 3.5)

# the line that does the actual plotting
FIG = sns.relplot(
                  data = df_filt, 
                  x = x_var, 
                  y = y_var, 
                  kind = 'line', 
                  hue = hue, 
                  # palette = sub_pal,
                  height = 12, 
                  aspect = 1.2, 
                  linewidth = 3, 
                  legend = 'full', 
                  facet_kws = {'legend_out':False}
                  )
# setting axis limits
FIG.set(xlim=xlim)
FIG.set(ylim=ylim)

plt.axvline(media_switch_frame*frame_interval_minutes)

# place legend
leg = plt.legend(loc='center',bbox_to_anchor=(1.5,0.5), frameon=False)
# set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(6)

# print number of cells (technically cell cycles)
types = list(set(df_filt[hue]))
num_types = len(types)
totcells = 0
for i in range(0,num_types):
    data_temp = df_filt.loc[df_filt[hue]==types[i]].copy()
    numcells = len(set(data_temp['cell_unique_id']))
    print(types[i] + ', n = ' + str(numcells))
    totcells = totcells + numcells
# USE SUM OF BOOLEANS FOR THE ABOVE

print('Total: N = ' + str(totcells))


# In[]: GET TIME-INDEPENDENT DATAFRAME

# select needed cols from overall_df_with_rel; time-independent fields only!
needed_cols = [
    'mutant', 'cell_unique_id', 'file', 'position', # identifying info
    'Whi5 exit to budding minutes', # should be small value; these should be well-correlated
    'Whi5 exit to media switch minutes', # independent variable
    'Budding to media switch minutes', # independent variable
    'Whi5 exit to reentry minutes', # dependent variable
    'Budding to Whi5 reentry minutes', # dependent variable
    'Whi5_reentry_frame',
    'Whi5_exit_frame'
]

# drop duplicates to get a dataframe where each cell only shows up once (i.e. no time-dependent traces)
df_filt_nodups = df_filt.loc[filter_idx, needed_cols].copy().drop_duplicates()
df_filt_nodups.mutant=pd.Categorical(df_filt_nodups.mutant,categories=mutants_to_plot) # ensures mutants are plotted in consistent order


# In[]: BAR/BOX/VIOLIN PLOTS

# HYPOTHESIS: Whi5-5S will re-enter nucleus faster/more readily than Whi5-WT

# need to control for time of exposure

x_var = 'mutant'

y_var = 'Whi5 exit to reentry minutes'

sns.set_theme(style='whitegrid', font_scale = 2.5)
FIG3 = sns.barplot(data = df_filt_nodups, palette = sub_pal, x = x_var, y = y_var) # can change to sns.barplot or sns.boxplot
sns.swarmplot(data = df_filt_nodups, x = x_var, y = y_var, color="0", size=15, alpha=0.5)
# FIG3.set(ylim=[0,None])
FIG3.figure.set_size_inches(12,8)

# show_values(FIG3, space = -10)


# In[]: VERSUS PLOTS

hue = 'mutant'

x_var = 'Whi5 exit to media switch minutes'
y_var = 'Whi5 exit to reentry minutes'

sns.set_theme(style='white', font_scale = 3.5)
FIGvs = sns.lmplot(data = df_filt_nodups, 
                    x = x_var, y = y_var, 
                    hue = hue, 
                    height = 12, aspect = 1.2, 
                    palette = sub_pal, 
                    # x_bins = 5,
                    x_jitter = 1.5,
                    y_jitter = 1.5,
                    line_kws={"lw":3}, scatter_kws={"s": 100}, legend='full', facet_kws={'legend_out':False})


# place legend
leg = plt.legend(loc='center',bbox_to_anchor=(1.08,0.5), frameon=False)
# set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(6)

# In[]: PRINT P VALUES

# List of unique mutants
mutants = mutants_to_plot
variable = 'Whi5 exit to reentry minutes'

# Perform pairwise t-tests and store p-values in a dictionary
p_values = {}

for i in range(len(mutants)):
    for j in range(i + 1, len(mutants)):
        mutant1 = mutants[i]
        mutant2 = mutants[j]
        
        data1 = df_filt_nodups[df_filt_nodups['mutant'] == mutant1][variable]
        data2 = df_filt_nodups[df_filt_nodups['mutant'] == mutant2][variable]
        
        # Perform t-test
        t_stat, p_val = ttest_ind(data1, data2)
        
        # Store the p-value in the dictionary with a label
        label = f'{mutant1} vs. {mutant2}'
        p_values[label] = p_val

# Print the p-values
for label, p_val in p_values.items():
    print(f'{label}: p-value = {p_val:.2e}')
    

