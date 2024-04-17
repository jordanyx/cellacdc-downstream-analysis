# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 15:41:09 2022

@author: jyxiao
"""

def calculate_phase_timing_quantities(df, frame_interval_minutes):
    # df must be overall_df_with_rel, containing all the fields used below
    # NOTE: this could probably be inserted into calculate_per_phase_quantities() from cca_functions.py, but not sure we want to mess with the acdc files
    
    # real experiment time
    df['time_i'] = df['frame_i'] * frame_interval_minutes
    
    # number of frames since the start of the current phase (G1 or S); starts at 1
    df['frames_in_phase'] = df['frame_i'] - df['phase_begin'] + 1
    df['minutes_in_phase'] = ( df['frames_in_phase'] - 1 ) * frame_interval_minutes # set to 0 for plotting
    
    # calculate the frames to the next (for G1 cells) and from the last (for S cells) G1/S transition
    # put another way, this is time until budding (for G1 cells) or time since budding (for S cells); time of budding = 0
    df['bud_aligned_frames_in_phase'] = df.apply(
        lambda x: x.loc['frames_in_phase'] -1 if x.loc['cell_cycle_stage']=='S' \
        #lambda x: x.loc['frames_in_phase'] if x.loc['cell_cycle_stage']=='S' \
        else x.loc['frames_in_phase'] -1 - x.loc['phase_length'],
        axis=1
    )
    # convert to minutes
    df['bud_aligned_time_in_minutes'] = df['bud_aligned_frames_in_phase'] * frame_interval_minutes
    
    # phase length from ACDC excludes starting frame, so we add it back in such that G1 + S lengths = total # frames in cell cycle
    df['phase_length_inclusive'] = df['phase_length'] + 1
    # convert to minutes
    df['phase_length_minutes'] = df['phase_length_inclusive'] * frame_interval_minutes
    
    # number of frames since the start of current cell cycle; starts at 1
    df['frames_in_cell_cycle'] = df.groupby(['Cell_ID', 'generation_num', 'position', 'file'])['frame_i'].transform('cumcount') + 1 # in frames
    df['minutes_in_cell_cycle'] = ( df['frames_in_cell_cycle'] - 1 ) * frame_interval_minutes 
    
    df['frame_interval_minutes'] = frame_interval_minutes
    
    return df.copy()

# =============================================================================
# def calculate_autofluo_corrected_quantities(df, ch_name, a, b):
#     # NOTE: may become unnecessary if autofluo calculations are moved to the ACDC GUI save step
#     # NEWER VERSION OF THIS FUNCTION IS BELOW
#     # df must be overall_df_with_rel, containing all the fields used below
#     # ch_name is channel to be corrected
#     # fit params (a and b) obtained from autofluo_analysis.ipynb
#     # this is a per-pixel fit, which we determined was better than the whole-cell fit
#     # a is in units of au/pix/vox
#     # b is in units of au/pix
#     # cell area is always in pixels
#     # use vox to determine autofluo correction
#     # use fL for concentration calculation
#     
#     # IMPORTANT NOTE: we subtract from a given channel's RAW total, ignoring ACDC's own background-corrected values.
#     
# # =============================================================================
# #     # JK'S LOWESS CORRECTION (replaces amount corrections)
# #     import pickle
# #     with open(r'C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\autof_lowess_func.pickle','rb') as pickle_in:
# #         lowess_func = pickle.load(pickle_in)
# #         autofluo_correction = lowess_func(df['cell_vol_vox'])
# #         autofluo_correction_rel = lowess_func(df['cell_vol_vox_downstream_rel'])
# #         
# #     df[f'{ch_name}_af_corrected_amount'] = df[f'{ch_name}_raw_sum'] - autofluo_correction # JK'S LOWESS CORRECTION
# #     df[f'{ch_name}_af_corrected_amount_rel'] = df[f'{ch_name}_raw_sum_rel'] - autofluo_correction_rel # JK'S LOWESS CORRECTION
# # =============================================================================
#     
#     # TESTING
#     # ch_name = 'VenusRaw' # REMEMBER TO ALSO CHANGE IN THE BELOW FUNCTION
#     # ch_name = 'mCitrineRaw'
#     # a = 8.868221835189484e-06
#     # b = df['mCitrineRaw_autoBkgr_bkgrVal_median']
#     
#     b = b + df[f'{ch_name}_autoBkgr_bkgrVal_median']
# 
#     # LINEAR AF-PER-PIXEL CORRECTION, for main cell and relative cell (i.e. associated bud or mother)
#     df[f'{ch_name}_af_corrected_amount'] = df[f'{ch_name}_raw_sum'] - df['cell_area_pxl'] * (b + a * df['cell_vol_vox'] )
#     df[f'{ch_name}_af_corrected_amount_rel'] = df[f'{ch_name}_raw_sum_rel'] - df['area_rel'] * (b + a * df['cell_vol_vox_downstream_rel'])
#     
#     # concentration is amount in au divided by vol in fL
#     df[f'{ch_name}_af_corrected_concentration'] = df[f'{ch_name}_af_corrected_amount'] / df['cell_vol_fl']
#     df[f'{ch_name}_af_corrected_concentration_rel'] = df[f'{ch_name}_af_corrected_amount_rel'] / df['cell_vol_fl_downstream_rel']
#     
#     # coefficient of variation over the area of the mother-bud-combined cell (proxy for nuclear localization)
#     df[f'{ch_name}_af_corrected_mean'] = df[f'{ch_name}_af_corrected_amount'] / df['cell_area_pxl']
#     df[f'{ch_name}_af_corrected_CV'] = df[f'{ch_name}_std'] / df[f'{ch_name}_af_corrected_mean']
#     # NOTE: standard deviation doesn't need to be AF-corrected, because the AF correction is the same subtraction for every pixel in the same cell
#     
#     return df.copy()
# =============================================================================

def calculate_autofluo_corrected_quantities(df, ch_name, a, b, piecewise_bool=False ,a_small=None, b_small=None, a_big=None, b_big=None):
    
    # NOTE: may become unnecessary if autofluo calculations are moved to the ACDC GUI save step
    
    # df must be overall_df_with_rel, containing all the fields used below
    # ch_name is channel to be corrected
    # fit params (a and b) obtained from autofluo_analysis.ipynb
    # this is a per-pixel fit, which we determined was better than the whole-cell fit
    # a is in units of au/pix/vox
    # b is in units of au/pix
    # cell area is always in pixels
    # use vox to determine autofluo correction
    # use fL for concentration calculation
    
    if piecewise_bool == False:
        # LINEAR AF-PER-PIXEL CORRECTION, for main cell and relative cell (i.e. associated bud or mother)
        # for each cell, the per-pixel correction is: intercept + slope * volume + median background pixel intensity
        # this per-pixel correction is multiplied by cell area for the total correction, which is subtracted from the channel's raw sum.
        
        df[f'{ch_name}_af_corrected_amount'] = df[f'{ch_name}_raw_sum'] - df['cell_area_pxl'] * (b + a * df['cell_vol_vox'] + df[f'{ch_name}_autoBkgr_bkgrVal_median'])
        df[f'{ch_name}_af_corrected_amount_rel'] = df[f'{ch_name}_raw_sum_rel'] - df['area_rel'] * (b + a * df['cell_vol_vox_downstream_rel'])
        
    else:
        df[f'{ch_name}_af_corrected_amount'] = df.apply(
            lambda x: x.loc[f'{ch_name}_raw_sum'] - x.loc['cell_area_pxl'] * (b_big + a_big * x.loc['cell_vol_vox'] + x.loc[f'{ch_name}_autoBkgr_bkgrVal_median']) if\
            x.loc['cell_vol_vox']>26773 else\
            x.loc[f'{ch_name}_raw_sum'] - x.loc['cell_area_pxl'] * (b_small + a_small * x.loc['cell_vol_vox'] + x.loc[f'{ch_name}_autoBkgr_bkgrVal_median']),
            axis=1
            )
        df[f'{ch_name}_af_corrected_amount_rel'] = df.apply(
            lambda x: x.loc[f'{ch_name}_raw_sum_rel'] - x.loc['cell_area_pxl_rel'] * (b_big + a_big * x.loc['cell_vol_vox_downstream_rel'] + x.loc[f'{ch_name}_autoBkgr_bkgrVal_median']) if\
            x.loc['cell_vol_vox']>26773 else\
            x.loc[f'{ch_name}_raw_sum_rel'] - x.loc['cell_area_pxl_rel'] * (b_small + a_small * x.loc['cell_vol_vox_downstream_rel'] + x.loc[f'{ch_name}_autoBkgr_bkgrVal_median']),
            axis=1
            )
    
    # concentration is amount in au divided by vol in fL
    df[f'{ch_name}_af_corrected_concentration'] = df[f'{ch_name}_af_corrected_amount'] / df['cell_vol_fl']
    df[f'{ch_name}_af_corrected_concentration_rel'] = df[f'{ch_name}_af_corrected_amount_rel'] / df['cell_vol_fl_downstream_rel']
    
    # coefficient of variation over the area of the mother-bud-combined cell (proxy for nuclear localization)
    df[f'{ch_name}_af_corrected_mean'] = df[f'{ch_name}_af_corrected_amount'] / df['cell_area_pxl']
    #df[f'{ch_name}_af_corrected_CV'] = df[f'{ch_name}_std'] / df[f'{ch_name}_af_corrected_mean']
    # NOTE: standard deviation doesn't need to be AF-corrected, because the AF correction is the same subtraction for every pixel in the same cell
    
    return df.copy()

def calculate_combined_mother_bud_quantities(df, ch_name):
    # NOTE: overall_df_with_rel already includes combined m/b quantities, but using a different background correction; this is our version
    
    # ch_name = 'VenusRaw' # REMEMBER TO ALSO CHANGE IN THE ABOVE FUNCTION
    # ch_name = 'mCitrineRaw' # REMEMBER TO ALSO CHANGE IN THE ABOVE FUNCTION
    
    # bud's contribution to total fluorescence
    df['Bud amount'] = df.apply(
        lambda x: x.loc[f'{ch_name}_af_corrected_amount_rel'] if  x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' \
        else 0,
        axis=1
    )
        
    # sum of mother and bud fluorescence
    df['Combined m&b amount'] = df.apply(
        lambda x: x.loc[f'{ch_name}_af_corrected_amount'] + x.loc[f'{ch_name}_af_corrected_amount_rel'] if x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' \
        else x.loc[f'{ch_name}_af_corrected_amount'],
        axis=1
    )
     
    # ratio of bud to mother volume; calculated using voxels, should be the same with fL
    df['Bud to mother volume ratio'] = df.apply(
        lambda x: x.loc['cell_vol_vox_rel'] / x.loc['cell_vol_vox'] if x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' \
        else 0,
        axis=1
    )        

    # ratio of bud to mother fluorescence amount
    df['Bud to mother amount ratio'] = df.apply(
        lambda x: x.loc[f'{ch_name}_af_corrected_amount_rel'] / x.loc[f'{ch_name}_af_corrected_amount'] if x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' \
        else 0,
        axis=1
    )
         
    #  total volume of mother plus bud in voxels
    df['Combined m&b volume vox'] = df.apply(
        lambda x: x.loc['cell_vol_vox'] + x.loc['cell_vol_vox_rel'] if\
        x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' else\
        x.loc['cell_vol_vox'],
        axis=1
    )
        
    # total volume of mother plus bud in femtoliters
    df['Combined m&b volume fL'] = df.apply(
        lambda x: x.loc['cell_vol_fl'] + x.loc['cell_vol_fl_rel'] if\
        x.loc['cell_cycle_stage']=='S' and x.loc['relationship'] == 'mother' else\
        x.loc['cell_vol_fl'],
        axis=1
    )

    # final concentration is total fluo in mother/bud pair divided by their combined volume
    df['Combined m&b concentration'] = df['Combined m&b amount'] / df['Combined m&b volume fL']
    
    return df.copy()


def calculate_complete_cycle_quantities(df, frame_interval_minutes):
    # calculates quantities that only make sense to consider for complete cell cycles
    import numpy as np
    
    complete_cycle_cell_ids = list(set( df[df['complete_cycle']==1]['cell_unique_id'] ))
    
    for cellID in complete_cycle_cell_ids:
        # real-time frame of cell cycle events; note "division" technically means the frame right before bud neck disappearance
        df.loc[ (df['cell_unique_id']==cellID), 'birth_frame' ] = \
            df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='G1') ] ['frame_i'].iat[0]
        df.loc[ (df['cell_unique_id']==cellID), 'budding_frame' ] = \
            df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='S' ) ] ['frame_i'].iat[0]
        df.loc[ (df['cell_unique_id']==cellID), 'division_frame' ] = \
            df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='S' ) ] ['frame_i'].iat[-1]
        
        # durations of cell cycle phases
        df.loc[ (df['cell_unique_id']==cellID), 'G1_total_frames' ] = \
            df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='G1') ] ['phase_length'].iat[0] + 1
        df.loc[ (df['cell_unique_id']==cellID), 'SG2M_total_frames' ] = \
            df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='S' ) ] ['phase_length'].iat[0] + 1
        df.loc[ (df['cell_unique_id']==cellID), 'cell_cycle_total_frames' ] = \
            df[ (df['cell_unique_id']==cellID) ] ['frames_in_cell_cycle'].max()
        
        # fluorescence quantities at birth, budding, division
        df.loc[ (df['cell_unique_id']==cellID), 'Combined m&b amount birth' ] = \
            df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='G1') ] ['Combined m&b amount'].iat[0]
        df.loc[ (df['cell_unique_id']==cellID), 'Combined m&b volume birth' ] = \
            df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='G1') ] ['Combined m&b volume fL'].iat[0]
        df.loc[ (df['cell_unique_id']==cellID), 'Combined m&b amount budding' ] = \
            df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='S' ) ] ['Combined m&b amount'].iat[0]
        df.loc[ (df['cell_unique_id']==cellID), 'Combined m&b volume budding' ] = \
            df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='S' ) ] ['Combined m&b volume fL'].iat[0]
        df.loc[ (df['cell_unique_id']==cellID), 'Combined m&b amount division' ] = \
            df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='S') ] ['Combined m&b amount'].iat[-1]
        df.loc[ (df['cell_unique_id']==cellID), 'Combined m&b volume division' ] = \
            df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='S') ] ['Combined m&b volume fL'].iat[-1]
        df.loc[ (df['cell_unique_id']==cellID), 'B to m volume ratio division' ] = \
            df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='S') ] ['Bud to mother volume ratio'].iat[-1]
        df.loc[ (df['cell_unique_id']==cellID), 'B to m amount ratio division' ] = \
            df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='S') ] ['Bud to mother amount ratio'].iat[-1]
        
        # for Whi5-mCitrine experiments with custom annotation; not all of them have this
        # 'Whi5_nuclear_exit' is 1 for rows where nuclear exit is annotated, otherwise 0
        if 'Whi5_nuclear_exit' in df.columns:
            if df[ (df['cell_unique_id']==cellID) & (df['Whi5_nuclear_exit']==1) ] ['frame_i'].empty: # make sure nuclear exit is recorded at the GUI level
                df.loc[ (df['cell_unique_id']==cellID) , 'Whi5_exit_frame' ] = np.nan # to fill the gap for cell cycles where exit isn't recorded
            else:
                # frame Whi5 leaves the nucleus
                df.loc[ (df['cell_unique_id']==cellID) , 'Whi5_exit_frame' ] = \
                    int( df[ (df['cell_unique_id']==cellID) & (df['Whi5_nuclear_exit']==1) ] ['frame_i'] )

                # time between Whi5 exit and budding; positive value means budding is after Whi5 exit (normal behavior)
                df.loc[ (df['cell_unique_id']==cellID) ,'Whi5 exit to budding frames' ] = \
                         df[ (df['cell_unique_id']==cellID) ] ['budding_frame'] - \
                         df[ (df['cell_unique_id']==cellID) ] ['Whi5_exit_frame']
                df['Whi5 exit to budding minutes'] = df['Whi5 exit to budding frames'] * frame_interval_minutes
            
            # 'Whi5_nuclear_reentry_after_starvation' is 1 for rows where nuclear reentry is annotated, otherwise 0
            if 'Whi5_nuclear_reentry_after_starvation' in df.columns:
                if df[ (df['cell_unique_id']==cellID) & (df['Whi5_nuclear_reentry_after_starvation']==1) ] ['frame_i'].empty: # make sure nuclear reentry is recorded at the GUI level
                    df.loc[ (df['cell_unique_id']==cellID) , 'Whi5_reentry_frame' ] = np.nan
                else:
                    # frame Whi5 re-enters the nucleus after starvation
                    df.loc[ (df['cell_unique_id']==cellID) , 'Whi5_reentry_frame' ] = \
                        int( df[ (df['cell_unique_id']==cellID) & (df['Whi5_nuclear_reentry_after_starvation']==1) ] ['frame_i'] )

                    # time between Whi5 exit and Whi5 reentry; this should be positive by definition
                    df.loc[ (df['cell_unique_id']==cellID) ,'Whi5 exit to reentry frames' ] = \
                             df[ (df['cell_unique_id']==cellID) ] ['Whi5_reentry_frame'] - \
                             df[ (df['cell_unique_id']==cellID) ] ['Whi5_exit_frame']
                    df['Whi5 exit to reentry minutes'] = df['Whi5 exit to reentry frames'] * frame_interval_minutes
                            
                    # time between budding and Whi5 reentry; this should be positive unless Whi5 somehow exits nucleus after budding, which shouldn't happen
                    df.loc[ (df['cell_unique_id']==cellID) ,'Budding to Whi5 reentry frames' ] = \
                             df[ (df['cell_unique_id']==cellID) ] ['Whi5_reentry_frame'] - \
                             df[ (df['cell_unique_id']==cellID) ] ['budding_frame']
                    df['Budding to Whi5 reentry minutes'] = df['Budding to Whi5 reentry frames'] * frame_interval_minutes
                    
        # for CLN2pr expression experiments; not meaningful for other things
        df.loc[ (df['cell_unique_id']==cellID), 'Conc increase SG2M peak minus budding' ] = \
            df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='S') ] ['Combined m&b concentration'].max() \
          - df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='G1') ] ['Combined m&b concentration'].iat[-1]
          
        # for CLN2pr expression experiments; not meaningful for other things
        df.loc[ (df['cell_unique_id']==cellID), 'Conc increase SG2M peak minus G1 trough' ] = \
            df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='S') ] ['Combined m&b concentration'].max() \
          - df[ (df['cell_unique_id']==cellID) & (df['cell_cycle_stage']=='G1') ] ['Combined m&b concentration'].min()
          
    
    # Note: these lengths are inclusive of the starting frames
    df['G1 length minutes'] = df['G1_total_frames'] * frame_interval_minutes
    df['SG2M length minutes'] = df['SG2M_total_frames'] * frame_interval_minutes
    df['Cell cycle length minutes'] = df['cell_cycle_total_frames'] * frame_interval_minutes
    
    df['Combined m&b conc birth'] = df['Combined m&b amount birth'] / df['Combined m&b volume birth']
    df['Combined m&b conc budding'] = df['Combined m&b amount budding'] / df['Combined m&b volume budding']
    df['Combined m&b conc division'] = df['Combined m&b amount division'] / df['Combined m&b volume division']

    df['G1 growth'] = df['Combined m&b volume budding'] - df['Combined m&b volume birth']
    df['SG2M growth'] = df['Combined m&b volume division'] - df['Combined m&b volume budding']
    df['Full cycle growth'] = df['Combined m&b volume division'] - df['Combined m&b volume birth']

    df['log birth size'] = np.log( df['Combined m&b volume birth'] )
    df['G1 relative growth'] = np.log(df['Combined m&b volume budding'] / df['Combined m&b volume birth']) #/ df_filt['Combined m&b volume birth']
    
    return df.copy()


def expo_decay_fit(xdata,a,b,c):
    # Used to fit exponential decay for certain experiments.
    import numpy as np
    return a * np.exp(xdata * b) + c



