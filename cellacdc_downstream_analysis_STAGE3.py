# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 14:09:18 2022

@author: jyxiao

ACDC OUTPUT ANALYSIS
STAGE 3: PLOTTING

Input: multiple CSVs from Stage 1 (CURRENTLY SKIPS STAGE 2)
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

overall_filepath = 'G:\My Drive\JX_DATA\Whi5-mCitrine expression\Whi5-mCitrine expression all/'
# overall_filepath = 'G:\My Drive\JX_DATA\CLN2pr expression\CLN2pr expression all/'

# overall_filepath = 'G:\My Drive\JX_DATA\Whi5-mCitrine expression\Jacob/'

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

# =============================================================================
# # In[]: VOLUME CHECK, for comparison to CC data
# 
# # take last frame as a snapshot
#  # exclude buds to avoid double-counting
# # filter_idx = (df_all['frame']==121)\
# #             &(df_all['generation_num']!=0)
# # df_lastframe = df_all.loc[filter_idx].copy()
# 
# mutants = list(set(df_all['mutant']))
# 
# for mutant in mutants:
#     mean_vol_fL = np.nanmean(df_all[ (df_all['mutant']==mutant)&(df_all['frame_i']==120)&(df_all['generation_num']!=0)] ['Combined m&b volume fL'] )
#     print(mutant + 'mean: ' + str(mean_vol_fL) + ' fL')
#     
# =============================================================================


# In[]: PRE-PLOTTING
# USE THIS ARRAY TO CHANGE WHICH MUTANTS GET PLOTTED
mutants_to_plot = [
                    'WT',    
                    '12A',   
                    # '12A x2', 
                    # '7A',    
                    # '7A x2',  
                    # '5S',    
                    # '19A', 
                    # '6A',
                    # '7T',    
                    'WT-NLS',
                    # '12A-NLS',
                    # 'WT-3mpf',
                    # '7A-3mpf',
                    # 'Whi5 del',    
                    # 'mutant error'
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
    "12A":   full_pal[9], # light blue
    "12A x2":full_pal[0], # dark blue
    "7A":    full_pal[3], # reddish orange
    "7A x2": full_pal[1], # lighter orange
    "5S":    full_pal[4], # light purple
    "19A":   full_pal[7], # grey
    "6A":    full_pal[6], # pink
    
    "WT-NLS":    full_pal[5], # light brown
    "12A-NLS":   full_pal[1], # lighter orange
    
    "WT-3mpf":   full_pal[9], # light blue
    "7A-3mpf":   full_pal[0], # dark blue
    
    "Whi5 del":  full_pal[8] # yellow
    }

sub_pal = [color_mapping[item] for item in mutants_to_plot]

    
# USER INPUT: filter for cells of interest
filter_idx = (df_all['complete_cycle']==1)         \
            &(df_all['generation_num']==1)         \
            &(df_all['to_plot']==True)         \
            &(df_all['file'] !='220928_JX_JX78c')    \
            &(df_all['file'] !='230202_JX_JX26Rb')    \
            &(df_all['file'] !='220721_JX_JX26a')    \
            &(df_all['file'] !='220710_JX_JX26a')    \
                # &(df_all['file'].str.contains('3mpf'))    \
            # &(df_all['file'] =='240124_JX_MS358_3mpf')    \
                
            # &(df_all['cell_cycle_stage']=='G1')     # NEED THIS LINE TO PLOT ONLY G1 DATA
            # &(df_all['file'] =='220709_JX_MS358_rev')    \
            # &(df_all['position'] == 'Position_1')           \
            
                
            # &(df_all['cell_unique_id'] == '220502_SM_JX61a_Position_1_Cell_2_Gen_1')        \
                # this line for one example cell
            
            # &(df_all['file'] !='230316_JX_JX101a')    \
                # &(df_all['file'] =='220502_SM_JX61a')    \
                    

df_filt = df_all.loc[filter_idx].copy()
df_filt.mutant = pd.Categorical(df_filt.mutant, categories=mutants_to_plot) # ensures mutants are plotted in consistent order


# In[]: FINAL CALCULATIONS (things that require compiled data)

if overall_filepath.__contains__('mCitrine'):
    # for Whi5-mCit: CoV normed to start at 1.0 (normalization values selected by eye)
    df_filt.loc[ (df_filt['mutant']=='WT'    ), 'CV normed' ] = \
        df_filt[ (df_filt['mutant']=='WT'    ) ] ['mCitrineScaledFFC_CV'] / 0.53
    df_filt.loc[ (df_filt['mutant']=='12A'   ), 'CV normed' ] = \
        df_filt[ (df_filt['mutant']=='12A'   ) ] ['mCitrineScaledFFC_CV'] / 0.49
    df_filt.loc[ (df_filt['mutant']=='12A x2'), 'CV normed' ] = \
        df_filt[ (df_filt['mutant']=='12A x2') ] ['mCitrineScaledFFC_CV'] / 0.61
    df_filt.loc[ (df_filt['mutant']=='7A'    ), 'CV normed' ] = \
        df_filt[ (df_filt['mutant']=='7A'    ) ] ['mCitrineScaledFFC_CV'] / 0.55
    df_filt.loc[ (df_filt['mutant']=='19A'   ), 'CV normed' ] = \
        df_filt[ (df_filt['mutant']=='19A'   ) ] ['mCitrineScaledFFC_CV'] / 0.57
    df_filt.loc[ (df_filt['mutant']=='5S'    ), 'CV normed' ] = \
        df_filt[ (df_filt['mutant']=='5S'    ) ] ['mCitrineScaledFFC_CV'] / 0.53
        
    df_filt.loc[ (df_filt['mutant']=='WT-NLS'    ), 'CV normed' ] = \
        df_filt[ (df_filt['mutant']=='WT-NLS'    ) ] ['mCitrineScaledFFC_CV'] / 0.50
    df_filt.loc[ (df_filt['mutant']=='12A-NLS'   ), 'CV normed' ] = \
        df_filt[ (df_filt['mutant']=='12A-NLS'   ) ] ['mCitrineScaledFFC_CV'] / 0.50
    
    # for Whi5-mCit: amounts rescaled to WT
    amount_mean = 9000 # selected by eye (WT amount at budding)
    df_filt['Combined m&b amount rescaled'] = df_filt['Combined m&b amount']/amount_mean
    
    # for Whi5-mCit: G1 concentrations normed to start at 1.0 (normalization values selected by eye)
    df_filt.loc[ (df_filt['mutant']=='WT'    ), 'G1 conc normed' ] = \
        df_filt[ (df_filt['mutant']=='WT'    ) ] ['Combined m&b concentration'] / 215
    df_filt.loc[ (df_filt['mutant']=='12A'   ), 'G1 conc normed' ] = \
        df_filt[ (df_filt['mutant']=='12A'   ) ] ['Combined m&b concentration'] / 87
    df_filt.loc[ (df_filt['mutant']=='12A x2'), 'G1 conc normed' ] = \
        df_filt[ (df_filt['mutant']=='12A x2') ] ['Combined m&b concentration'] / 152
    df_filt.loc[ (df_filt['mutant']=='7A'    ), 'G1 conc normed' ] = \
        df_filt[ (df_filt['mutant']=='7A'    ) ] ['Combined m&b concentration'] / 150
    df_filt.loc[ (df_filt['mutant']=='19A'   ), 'G1 conc normed' ] = \
        df_filt[ (df_filt['mutant']=='19A'   ) ] ['Combined m&b concentration'] / 118
    df_filt.loc[ (df_filt['mutant']=='5S'    ), 'G1 conc normed' ] = \
        df_filt[ (df_filt['mutant']=='5S'    ) ] ['Combined m&b concentration'] / 210

elif overall_filepath.__contains__('CLN2pr'):
    # for CLN2pr-Venus: concentrations normed to WT
    conc_mean = 155 # selected by eye (WT concentration at budding)
    df_filt['Combined m&b concentration rescaled'] = df_filt['Combined m&b concentration']/conc_mean
    
    conc_diff_mean = 312
    df_filt['Conc diff rescaled'] = df_filt['Conc increase SG2M peak minus budding']/conc_diff_mean

# minutes in phase; has also been moved to utils, but have to rerun Stage 1 on every movie first
df_filt['minutes_in_phase'] = df_filt['frames_in_phase'] * 6 - 6 


# In[]: TIME-DEPENDENT PLOTTING

# choose fields to plot
# x_var = 'minutes_in_phase'
x_var = 'bud_aligned_time_in_minutes'

# y_var = 'mCitrineScaledFFC_CV'
# y_var = 'CV normed'

# y_var = 'Bud to mother volume ratio'
# y_var = 'Bud to mother amount ratio'

# y_var = 'Combined m&b volume fL'

# y_var = 'Combined m&b amount'
y_var = 'Combined m&b amount rescaled'
# y_var = 'G1 conc normed'

# y_var = 'Combined m&b concentration'
# y_var = 'Combined m&b concentration rescaled'
# y_var = 'cell_vol_vox'

xlim = [-120,120] # for whole cycle
# xlim = [-80,120] # for Whi5-mCit CV
# xlim = [-60,120] # for CLN2 stuff
# xlim = [0,120] # for G1

# ylim = [0.7,1.1] # for Whi5-mCit normed CV
ylim = [0,2.5] # for Whi5-mCit rescaled amounts
# ylim = [0, None] # everything else

# overlaying by mutant, position, etc.
# hue = 'file'
hue = 'mutant'
# hue = 'position'
# hue = 'cell_unique_id'

sns.set_theme(style='white', font_scale = 3.5)

# the line that does the actual plotting
FIG = sns.relplot(
                  data = df_filt, 
                  x = x_var, 
                  y = y_var, 
                  kind = 'line', 
                  hue = hue, 
                   palette = sub_pal,
                  height = 12, 
                  aspect = 1.2, 
                  linewidth = 3, 
                  legend = 'full', 
                  facet_kws = {'legend_out':False}
                  )
# setting axis limits
FIG.set(xlim=xlim)
FIG.set(ylim=ylim)

# plot 1/V curve
if y_var == 'G1 conc normed':
    x_arr = np.arange(xlim[0],xlim[1]+1)
    y_arr = 0.5**(x_arr/150) # doubling time of 150 min
    plt.gcf()
    plt.plot(x_arr,y_arr,'k', linewidth = 3)

# place legend
leg = plt.legend(loc='center',bbox_to_anchor=(1.2,0.5), frameon=False)
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

# title and axis labels
#FIG.fig.suptitle('Whi5-mCitrine localization')
#FIG.fig.suptitle('Whi5-mCitrine localization')

# FIG.axes[0,0].set_xlabel('Time aligned to budding (min)')
#FIG.axes[0,0].set_ylabel(y_var)
# FIG.axes[0,0].set_ylabel('Coefficient of variation')
# FIG.axes[0,0].set_ylabel('Whi5-mCitrine amount (a.u.)')
# FIG.axes[0,0].set_ylabel('Whi5-mCitrine concentration (a.u.)')
# FIG.axes[0,0].set_ylabel('mVenus-PEST conc. (a.u.)')

# FIG.savefig(path.join(overall_filepath,'\Whi5-mCitrine_expression_all.png'))

# In[]: SIZE VERSUS WHI5 CONCENTRATION SCATTERPLOT

x_var = 'Combined m&b volume fL'
y_var = 'Combined m&b concentration'
num_edges = 8 # number of bins is num_edges minus 1
x_range = [20,100]
binsize = int( (x_range[1]-x_range[0]) / num_edges)

bin_edges = [i for i in range(x_range[0], x_range[1], binsize)]
df_filt['Volume bin'] = pd.cut(df_filt[x_var], bins=bin_edges, include_lowest=True, labels=False)

hue = 'mutant'

sns.set_theme(style='white', font_scale = 2.5)
FIGscat = sns.lineplot(
                  data = df_filt, 
                  x = 'Volume bin',
                  y = y_var, 
                  # kind = 'line', 
                  hue = hue, 
                  palette = sub_pal, 
                  # height = 12, 
                  # aspect = 1.2, 
                  linewidth = 3, 
                  ci = 95
                  )


needed_cols = ['mutant', 'cell_unique_id', 'file', 'position', # identifying info
               'Combined m&b volume birth','Combined m&b conc birth']
df_filt_nodups = df_filt.loc[filter_idx, needed_cols].copy().drop_duplicates()
df_filt_nodups['Volume bin'] = pd.cut(df_filt[x_var], bins=bin_edges, include_lowest=True, labels=False)

sns.lineplot(
                  data = df_filt_nodups, 
                  x = 'Volume bin', 
                  y = 'Combined m&b conc birth', 
                  hue = hue,
                  linewidth = 3, 
                  ci = 95,
                  ax = FIGscat
                  )


# Create a mapping between binned values and original x-values
bin_to_original = {i: bin_edges[i] for i in range(len(bin_edges) - 1)}

# Replace the x-axis tick labels with the original x-values
FIGscat.set_xticks(range(len(bin_edges) - 1))
FIGscat.set_xticklabels([bin_to_original[i] for i in range(len(bin_edges) - 1)])

# setting axis limits
# FIGscat.set(xlim=[-1, 10])
FIGscat.set(ylim=[0, 320])
FIGscat.figure.set_size_inches(12,10)
FIGscat.set(title="Whi5[WT]-mCitrine cells in G1")


# In[]: GET TIME-INDEPENDENT DATAFRAME

# select needed cols from overall_df_with_rel; time-independent fields only!
needed_cols = [
    'mutant', 'cell_unique_id', 'file', 'position', # identifying info
    'G1_total_frames', 'SG2M_total_frames', 'cell_cycle_total_frames', # phase lengths
    'G1 length minutes', 'SG2M length minutes', 'Cell cycle length minutes', # phase lengths
    'Combined m&b volume birth', 'Combined m&b volume budding', 'Combined m&b volume division', # volumes
    'Combined m&b amount birth', 'Combined m&b amount budding', 'Combined m&b amount division', # fluo amounts
    'Combined m&b conc birth', 'Combined m&b conc budding', 'Combined m&b conc division', # concentrations
    'B to m volume ratio division', 'B to m amount ratio division', # bud to mother ratios
    'G1 growth', 'SG2M growth', 'Full cycle growth', # volume growth
    'log birth size', 'G1 relative growth',
    'Whi5 exit to budding minutes', # only used for Whi5-mCitrine experiments
    # 'Conc increase SG2M peak minus budding', 'Conc diff rescaled' # only used for CLN2pr experiments
]

# drop duplicates to get a dataframe where each cell only shows up once (i.e. no time-dependent traces)
df_filt_nodups = df_filt.loc[filter_idx, needed_cols].copy().drop_duplicates()
df_filt_nodups.mutant=pd.Categorical(df_filt_nodups.mutant,categories=mutants_to_plot) # ensures mutants are plotted in consistent order


# In[]: BAR/BOX/VIOLIN PLOTS

x_var = 'mutant'
# y_var = 'G1 length minutes'
# y_var = 'SG2M length minutes'
# y_var = 'Cell cycle length minutes'
# y_var = 'Combined m&b volume birth'
# y_var = 'Combined m&b volume budding'
# y_var = 'Combined m&b volume division'

y_var = 'B to m volume ratio division'
# y_var = 'B to m amount ratio division'

# y_var = 'G1 growth'
# y_var = 'SG2M growth'
# y_var = 'Full cycle growth'

# y_var = 'SG2M growth'

# y_var = 'Conc increase SG2M peak minus budding'
# y_var = 'Conc diff rescaled'

# y_var = 'Whi5 exit to budding minutes'

sns.set_theme(style='whitegrid', font_scale = 2.5)
FIG3 = sns.barplot(data = df_filt_nodups, palette = sub_pal, x = x_var, y = y_var) # can change to sns.barplot or sns.boxplot
# sns.swarmplot(data = df_filt_nodups, x = x_var, y = y_var, color="0", size=15, alpha=0.5)
# FIG3.set(ylim=[0,None])
FIG3.figure.set_size_inches(12,8)

# show_values(FIG3, space = -10)

# In[]: VERSUS PLOTS

hue = 'mutant'

x_var = 'Combined m&b volume birth'
y_var = 'G1 growth'
# y_var = 'G1 relative growth'

# x_var = 'Combined m&b volume budding'
# y_var = 'SG2M growth'

# x_var = 'Combined m&b volume birth'
# x_var = 'Combined m&b volume budding'
# y_var = 'G1 length minutes'
# y_var = 'SG2M length minutes'
# y_var = 'Cell cycle length minutes'

# x_var = 'log birth size'
# y_var = 'G1 relative growth'

# x_var = 'Combined m&b volume birth'
# y_var = 'Combined m&b conc birth'

# x_var = 'G1 length minutes'
# y_var = 'G1 relative growth'
# df['G1 relative growth'] = np.log(df['Combined m&b volume budding'] / df['Combined m&b volume birth'])

# df_filt_nodups['SG2M relative growth'] = np.log(df_filt_nodups['Combined m&b volume division'] / df_filt_nodups['Combined m&b volume budding'])
# x_var = 'SG2M length minutes'
# y_var = 'SG2M relative growth'

# x_var = 'Combined m&b volume birth'
# y_var = 'Combined m&b volume budding'

sns.set_theme(style='white', font_scale = 3.5)
FIGvs = sns.lmplot(data = df_filt_nodups, 
                    x = x_var, y = y_var, 
                    hue = hue, 
                    height=12, aspect=1.2, 
                    palette = sub_pal, 
                    x_bins = 5,
                    line_kws={"lw":3}, scatter_kws={"s": 100}, legend='full', facet_kws={'legend_out':False})

# =============================================================================
# FIGvs = sns.scatterplot(data = df_filt_nodups, 
#                    x = x_var, y = y_var, 
#                    hue = hue, 
#                    # height=12, aspect=1.2, 
#                    # palette = sub_pal, 
#                     x_bins = 6,
#                    # line_kws={"lw":3}, scatter_kws={"s": 100}, legend='full', facet_kws={'legend_out':False})
#     )
# =============================================================================
# axis limits
FIGvs.set(xlim=(20, 80), ylim=(0, 50)) # size vs growth
# FIGvs.set(xlim=(20, 80), ylim=(0, 170)) # phase lengths
# FIGvs.set(xlim=(3.1, 4.5), ylim=(0, 1.0)) # log size vs rel growth

# place legend
leg = plt.legend(loc='center',bbox_to_anchor=(1.13,0.5), frameon=False)
# set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(6)

# FIGvs.set(xlim=(0, 200), ylim=(0, 1.1)) # size vs growth


# In[]: FACETGRID PLOT FOR ALL MUTANTS
# makes one plot per mutant for the specified axes, also prints regression slope

# x_var = 'Combined m&b volume birth'
# y_var = 'G1 growth'
# y_var = 'G1 length minutes'

x_var = 'Combined m&b volume budding'
# y_var = 'SG2M growth'
y_var = 'SG2M length minutes'

sns.set_theme(style='white', font_scale = 3.5)
FIGgrid = sns.lmplot(data = df_filt_nodups, 
           x = x_var, 
           y = y_var, 
           col = hue, 
           palette = sub_pal,
           height=12, aspect=0.8, )

# Iterate over each subplot and add slope annotation
counter = 0
for ax in FIGgrid.axes.flat:
    mutant = mutants_to_plot[counter]
    slope, intercept, r_value, p_value, std_err = linregress(df_filt_nodups.loc[df_filt_nodups['mutant']==mutant][x_var], 
                                                             df_filt_nodups.loc[df_filt_nodups['mutant']==mutant][y_var])
    ax.annotate(f'Slope: {slope:.2f}', xy=(0.6, 0.85), xycoords='axes fraction')
    counter = counter + 1


# In[]: CDF PLOTS
# x_var = 'G1 length minutes'
# x_var = 'SG2M length minutes'
# x_var = 'Cell cycle length minutes'
x_var = 'Combined m&b volume birth'
# x_var = 'Combined m&b volume budding'
# x_var = 'Combined m&b volume division'
# x_var = 'G1 growth'
# x_var = 'SG2M growth'
# x_var = 'Full cycle growth'

# x_var = 'G1 relative growth'

hue = 'mutant'

sns.set(rc={'figure.figsize':(8,8)})

sns.set_theme(style='white', font_scale = 2)
FIG_CDF = sns.ecdfplot(data=df_filt_nodups, x=x_var, hue=hue, palette = sub_pal)

# FIG_CDF.axes.set_xlabel('Volume at birth (fL)')
# FIG_CDF.axes.set_xlabel('Volume at budding (fL)')
# FIG_CDF.axes.set_xlabel('Volume at division (fL)')
# FIG_CDF.axes.set_xlabel('G1 length (minutes)')
# FIG_CDF.axes.set_xlabel('S/G2/M length (minutes)')
# FIG_CDF.axes.set_xlabel('Full cycle length (minutes)')

# FIG_CDF.axes.set_xlim([0,300])

# In[]: PRINT P VALUES

# List of unique mutants
mutants = mutants_to_plot
variable = 'B to m volume ratio division'

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
    

# In[]: COMPARING SPECIFIC SIZE BINS

# filter for mutant and cell size

mutants_to_filter = ['WT', '7A', 'WT-3mpf', '7A-3mpf']

# Define the desired range of sizes at birth
Vmin = 44
Vmax = 50

# Create the filter criteria
mutant_criteria =  df_filt_nodups['mutant'].isin(mutants_to_filter)
size_criteria =   (df_filt_nodups['Combined m&b volume birth'] >= Vmin)  \
                & (df_filt_nodups['Combined m&b volume birth'] <= Vmax)

# Use the |? operator to combine the mutant and size range criteria
df_filt_again = df_filt_nodups[mutant_criteria & size_criteria]

# y_var = 'G1 growth'
y_var = 'G1 length minutes'

sns.set( style="whitegrid", font_scale = 3.5 )
fig, ax = plt.subplots(figsize=[20,12])

bars = sns.barplot(data = df_filt_again, x="mutant", y=y_var, capsize=.1, palette = sub_pal)#ci="0.95")
sns.swarmplot(data = df_filt_again, x="mutant", y=y_var, color="0", size=15, alpha=0.5)

show_values(bars, space = -60)
plt.title(f'Cells between {Vmin} and {Vmax} fL at birth')

# get p values
mutants = mutants_to_filter
variable = y_var

# Perform pairwise t-tests and store p-values in a dictionary
p_values = {}
for i in range(len(mutants)):
    for j in range(i + 1, len(mutants)):
        mutant1 = mutants[i]
        mutant2 = mutants[j]
        
        data1 = df_filt_again[df_filt_again['mutant'] == mutant1][variable]
        data2 = df_filt_again[df_filt_again['mutant'] == mutant2][variable]
        
        # Perform t-test
        t_stat, p_val = ttest_ind(data1, data2)
        
        # Store the p-value in the dictionary with a label
        label = f'{mutant1} vs. {mutant2}'
        p_values[label] = p_val

# Print the p-values
for label, p_val in p_values.items():
    print(f'{label}: p-value = {p_val:.2e}')


# In[]: CONDITIONAL PROBABILITY TESTING (CAUSAL INFERENCE)

hue = 'mutant'

# x_var = 'Combined m&b volume birth'
# y_var = 'Combined m&b volume budding'
# z_var = 'Combined m&b volume division'

x_var = 'Combined m&b volume birth'
y_var = 'Combined m&b conc birth'
z_var = 'G1 length minutes'

mutant = 'WT'

x = df_filt_nodups.loc[df_filt_nodups['mutant']==mutant][x_var]
y = df_filt_nodups.loc[df_filt_nodups['mutant']==mutant][y_var]
z = df_filt_nodups.loc[df_filt_nodups['mutant']==mutant][z_var]

# assuming y = ax + b
a, b, r_xy, p_value, std_err = linregress(x,y)
r_xy_squared = r_xy**2
print(f'y = {a:.2f} * x + {b:.2f}, r^2 = {r_xy_squared:.6f}')

# assuming z = cy + d
c, d, r_yz, p_value, std_err = linregress(y,z)
r_yz_squared = r_yz**2
print(f'z = {c:.2f} * y + {d:.2f}, r^2 = {r_yz_squared:.6f}')

x_condy = x - (y - b)/a
z_condy = z - c*y - d

# before conditional on y
slope1, intercept1, r1, p_value, std_err = linregress(x,z)
r1_squared= r1**2
print(f'z = {slope1:.2f} * x + {intercept1:.2f}, r^2 = {r1_squared:.2f}')
FIGtest = sns.regplot(x,z,)

# after conditional on y
slope2, intercept2, r2, p_value, std_err = linregress(x_condy,z_condy)
r2_squared= r2**2
print(f'z_condy = {slope2:.2f} * x_condy + {intercept2:.2f}, r^2 = {r2_squared:.2f}')
plt.figure()

FIGtest2 = sns.regplot(x_condy,z_condy,)

#FIGtest.set(title = mutant)


