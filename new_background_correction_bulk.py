# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:19:28 2023

@author: Jacob
"""

'''
- make a list of base directories
- grab positions, img prefix,
- check each position has acdc output file
- check each position has mCitrine file
'''
#%% packages

import numpy as np
import tifffile as tf
import pandas as pd
from os import path
from glob import glob
import re

#%% Add your directories and foldernames here

root_dir = (
    #r'G:\My Drive\Whi5pr_mutants'
    # r'G:\My Drive\JX_DATA\CLN2pr expression'
    r'G:/My Drive/JX_DATA/Whi5-mCitrine shutoff'
    #this is the folder where all the acdc expt folders are located
    )

expt_names = [ #list of experiment folder names. Should contain the acdc Position_# folders
              r'220630_JX_MS380_2ng-uL-aTc_4hr-on_8hr-off',
              r'220723_JX_MS380_2ng-uL-aTc_6hr-on_6hr-off',
              r'220827_JX_JX73a_2ng-uL-aTc_6hr-on_6hr-off',
              r'220829_JX_MS380_2ng-uL-aTc_6hron_6hroff_12minperframe',
              r'220918_JX_JX73a_2ng-uL-aTc_6hron_6hroff_12minperframe',
              r'220929_JX_MS380_2ng-uL-aTc_12hroff_24minperframe',
              r'221012_JX_MS380_0aTc_24minperframe_AF',
              r'221101_JX_JX73a_2ng-uL-aTc_12hroff_24minperframe',
              
    # r'220502_SM_JX61a',
    # r'220505_JX_JX64a',
    # r'220515_JX_JX62c',
    # r'220516_JX_JX63b',
    # r'220522_JX_JX50b',
    # r'220522_JX_JX63a',
    # r'220523_JX_JX69a',
    # r'220608_JX_JX61b',
    # r'220609_JX_JX62b',
    # r'220610_JX_JX61a',
    # r'220610_JX_JX62c',
    # r'220611_JX_JX63a',
    # r'220613_JX_JX63b',
    # r'220616_JX_JX61b',
    # r'220617_JX_JX62b',
    # r'220626_JX_JX64a',
    # r'220712_JX_JX68a',
    # r'220713_JX_JX68b',
    # r'220928_JX_JX78c_BADSTRAIN',
    # r'221001_JX_JX78a_BADSTRAIN',
    # r'230128_JX_JX99a_data_good',
    # r'230211_JX_JX99c',
    ]

ff_file = (
    r"C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\Flat and dark fields\230210_scaled_net_flatfield.npy" 
    #path of scaled net flatfield image (ff-df/mean(ff-df))
    )

df_file = (
    r"C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\Flat and dark fields\230210_DFcombined_mean.tif" 
    #path of flatfield median image
    )

ch_name = 'mCitrine'
#%% 
expt_names.sort()
expt_num = len(expt_names)

# this section creates a flatfield image with 
ff_img = np.load(ff_file)
df_img = tf.imread(df_file).astype(float)

# =============================================================================
# ff = np.empty((1,121,1,1028,1216),dtype = 'float')
# for t in range(121):
#     ff[0,t,0,:,:] = ff_img
# =============================================================================

#%% load image data

for n in range(expt_num): #for each expt
    expt_dir = path.join(root_dir,expt_names[n]) #get expt name
    # img_file = czi_list[n] #get czi name
    # img_czi = CziFile(img_file) #load czi file
    pos_dir_list = glob(path.join(expt_dir,'Position_*','Images')) #this gets a list of paths to open the individual positions
    for pos_dir in pos_dir_list: #for each position
        temp_filename = path.basename(glob(path.join(pos_dir,'*metadata.csv'))[0]) #grab a random file so that we can extract the prefix
        file_prefix = temp_filename.replace('metadata.csv','') #delete the file type so that we're left with the prefix
        temp_posname = re.search(r'Position_[0-9]+',pos_dir).group(0) #regex search for czi position number "_pos#_"
        pos_num = int(temp_posname.split('_')[1]) #isolate just the position number
        
        if path.exists(path.join(pos_dir,file_prefix+ch_name+'ScaledFFC_aligned.npz')):
            print(f'position {pos_num} already fixed')
        else:
            img_file = (
                path.join(pos_dir,file_prefix+f'{ch_name}Raw.tif')
                )
            raw_mCit = tf.imread(img_file).astype(float)
            
            nframes = raw_mCit.shape[0]
            
            
            if path.exists(path.join(pos_dir,file_prefix+'align_shift.npy')):
                bool_align = True
                align_coords = np.load( #grabs the align shift coordinate file
                    path.join(pos_dir,file_prefix+'align_shift.npy')
                    )
            else:
                align_coords = np.zeros((nframes,2))
            
            
            crop_bound_file = ( #grabs the crop boundary file
                path.join(pos_dir,file_prefix+'dataPrepROIs_coords.csv')
                )
            
            crop_bounds = pd.read_csv(crop_bound_file) # open crop boundary file
            crop_bounds_values = np.array(crop_bounds["value"])
            if crop_bounds_values[4]:
                None
                bool_crop = True #lol bool crop sounds like bull crap
            else:
                crop_bounds_values = np.array([
                    0,
                    raw_mCit.shape[2],
                    0,
                    raw_mCit.shape[1],
                    0
                    ])
                bool_crop = False
            crop_bounds_array = np.array([crop_bounds_values[0],crop_bounds_values[1],
                                          crop_bounds_values[2],crop_bounds_values[3]]) #sets crop boundary as array for later use
            corrected_img = np.zeros(shape=[nframes,
                                            1,
                                            1,
                                            crop_bounds_values[3]-crop_bounds_values[2],
                                            crop_bounds_values[1]-crop_bounds_values[0],
                                            1],dtype='int16') #initiates the corrected image by creating empty array in the correct shape
                
            #aligned_crop_bounds = np.empty(shape=[nframes,4],dtype='int64') #initiates new array of crop boundaries with proper alignment shift
            
            for t in range(nframes): 
                temp_align_coords = np.array([align_coords[t,1],align_coords[t,1],
                                              align_coords[t,0],align_coords[t,0]],dtype = int) #arranges align coordinates to subtract from crop boundaries
                aligned_crop_bounds = crop_bounds_array - temp_align_coords #aligns crop boundaries
                aligned_crop_bounds[aligned_crop_bounds<0] = 0
                # temp_raw_mCit = raw_mCit
                
                ###make it into reading Raw ch image move to outside for loop
                
                # temp_raw_mCit = np.squeeze(temp_raw_mCit)
                temp_left = aligned_crop_bounds[0]
                temp_right = aligned_crop_bounds[1]
                temp_top = aligned_crop_bounds[2]
                temp_bottom = aligned_crop_bounds[3]
                if bool_crop:
                    corrected_img[t,0,0,:,:,0] = ( #subtract new flatfield from raw mCitr image in the aligned crop boundaries
                        (
                            raw_mCit[t,:,:] 
                            - df_img[temp_top:temp_bottom,temp_left:temp_right]
                        )
                        / ff_img[temp_top:temp_bottom,temp_left:temp_right]
                    )
                else:
                    corrected_img[t,0,0,temp_top:temp_bottom,temp_left:temp_right,0] = ( #subtract new flatfield from raw mCitr image in the aligned crop boundaries
                        (
                            raw_mCit[t,temp_top:temp_bottom,temp_left:temp_right] 
                            - df_img[temp_top:temp_bottom,temp_left:temp_right]
                        )
                        / ff_img[temp_top:temp_bottom,temp_left:temp_right]
                    )
                corrected_img[corrected_img < 0] = 0 # no negative numbers
            print('finished making corrected images for position ' + str(pos_num))
    
            #corrects the metatadata
            metadata_df = pd.read_csv(path.join(pos_dir,file_prefix+'metadata.csv')) #read metadata
            max_ch_num = int(max(metadata_df['Description'].str.findall('[0-9]'))[0])#find max channel number
            channel_dict = { #establish channel name and wavelength to append to metadata
                'Description':[f'channel_{max_ch_num + 1}_name', f'channel_{max_ch_num + 1}_emWavelen'],
                'values':[ch_name + 'ScaledFFC', 0.0]
               }
            channel_df = pd.DataFrame(channel_dict) #turn this into dataframe to concatenate later
            
            metadata_output = pd.concat([metadata_df,channel_df],ignore_index = True) #append new rows to dataframe
            metadata_output.to_csv(path.join(pos_dir,file_prefix+'metadata.csv'), index=False) #save metadata
            
            #writes new ffc files
            tf.imwrite(path.join(pos_dir,file_prefix+ch_name+'ScaledFFC.tif'),
                             corrected_img, imagej=True, metadata={'axes': 'TZCYXS'})
            np.savez(path.join(pos_dir,file_prefix+ch_name+'ScaledFFC_aligned.npz'), np.squeeze(corrected_img))
            print('saved position ' + str(pos_num))
    print('finished_'+expt_names[n])

    
