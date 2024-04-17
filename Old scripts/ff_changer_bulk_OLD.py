# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:49:17 2022
@author: Jacob Kim
runs ff_changer on every position of an experiment
"""


'''
- make a list of base directories
- grab positions, img prefix,
- check each position has acdc output file
- check each position has mCitrine file

Note by JX:
This script was written to account for a mistake made that only applies to movies analyzed between April and June 2022.
It should not need to be run more than once per user.

'''
#%% packages

from aicsimageio import AICSImage
from aicspylibczi import CziFile
import numpy as np
import tifffile
import pandas as pd
from os import path
import os
from glob import glob
import re
#%% Add your directories and foldernames here

root_dir = (
    r'G:\My Drive\JX_DATA\CLN2pr expression\TO_CORRECT' #r'G:\My Drive\Whi5pr_mutants' #this is the folder where all the acdc expt folders are located
    )

expt_names = [ #list of experiment folder names. Should contain the acdc Position_# folders
# =============================================================================
#     r'220505_JX_JX64a',
#     r'220515_JX_JX62c',
#     r'220516_JX_JX63b',
#     r'220522_JX_JX50b',
#     r'220522_JX_JX63a',
#     r'220523_JX_JX69a',
#     r'220608_JX_JX61b',
#     r'220609_JX_JX62b',
#     r'220610_JX_JX61a',
#     r'220610_JX_JX62c',
#     r'220611_JX_JX63a',
#     r'220613_JX_JX63b',
#     r'220616_JX_JX61b',
#     r'220617_JX_JX62b'
# =============================================================================
      r'220617_JX_JX62b'
    ]

czi_dir = (
    r'G:\My Drive\JX_DATA\CLN2pr expression\czi_files' # this is the folder where all czi files are located. Must be only czi files and nothing else
    # this should have the same dates as the expt folders, so that both can be sorted and referenced back to each other
    )

ff_file = (
    r"C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\AxioCam705_mCitrine_ff - new.tif" # path of flatfield median image
    )

ch_name = 'Venus'
#%% 
expt_names.sort()
expt_num = len(expt_names)
czi_list = glob(path.join(czi_dir,'*.czi'))
czi_list.sort() 

# this section creates a flatfield image with 
ff_img = AICSImage(ff_file)
ff = np.empty((1,121,1,1028,1216),dtype = 'float')
temp_ff = ff_img.data
temp_ff = temp_ff[0,0,0,:,:]
temp_ff = temp_ff.astype(float)
for t in range(121):
    ff[0,t,0,:,:] = temp_ff

#%% load image data

for n in range(expt_num): #for each expt
    expt_dir = path.join(root_dir,expt_names[n]) #get expt name
    img_file = czi_list[n] #get czi name
    img_czi = CziFile(img_file) #load czi file
    pos_dir_list = glob(path.join(expt_dir,'Position_*','Images')) #this gets a list of paths to open the individual positions
    for pos_dir in pos_dir_list: #for each position
        temp_filename = path.basename(glob(path.join(pos_dir,'*metadata.csv'))[0]) #grab a random file so that we can extract the prefix
        file_prefix = temp_filename.replace('metadata.csv','') #delete the file type so that we're left with the prefix
        temp_posname = re.search(r'_pos[0-9]+_',temp_filename).group(0) #regex search for czi position number "_pos#_"
        czi_pos_num = int(re.search(r'[0-9]+',temp_posname).group(0)) #isolate just the position number
        
        align_coords = np.load( #grabs the align shift coordinate file
            path.join(pos_dir,file_prefix+'align_shift.npy')
            )
        crop_bound_file = ( #grabs the crop boundary file
            path.join(pos_dir,file_prefix+'dataPrepROIs_coords.csv')
            )
        nframes = len(align_coords) #number of frames
        crop_bounds = pd.read_csv(crop_bound_file) # open crop boundary file
        crop_bounds_values = crop_bounds["value"]
        crop_bounds_array = np.array([crop_bounds_values[0],crop_bounds_values[1],
                                      crop_bounds_values[2],crop_bounds_values[3]]) #sets crop boundary as array for later use
        corrected_img = np.empty(shape=[nframes,
                                        1,
                                        1,
                                        crop_bounds_values[3]-crop_bounds_values[2],
                                        crop_bounds_values[1]-crop_bounds_values[0],
                                        1],dtype='int16') #initiates the corrected image by creating empty array in the correct shape
        cropped_raw_img = np.empty(shape=[nframes,
                                        1,
                                        1,
                                        crop_bounds_values[3]-crop_bounds_values[2],
                                        crop_bounds_values[1]-crop_bounds_values[0],
                                        1],dtype='int16')
        aligned_crop_bounds = np.empty(shape=[nframes,4],dtype='int64') #initiates new array of crop boundaries with proper alignment shift
        
        for t in range(nframes): 
            temp_align_coords = np.array([align_coords[t,1],align_coords[t,1],
                                          align_coords[t,0],align_coords[t,0]]) #arranges align coordinates to subtract from crop boundaries
            aligned_crop_bounds[t] = crop_bounds_array - temp_align_coords #aligns crop boundaries
            temp_raw_mCit,temp_shp = img_czi.read_image(S=czi_pos_num-1, C=1, T=t) #read specific frame of raw mCit image
            temp_raw_mCit = np.squeeze(temp_raw_mCit)
            temp_left = aligned_crop_bounds[t][0]
            temp_right = aligned_crop_bounds[t][1]
            temp_top = aligned_crop_bounds[t][2]
            temp_bottom = aligned_crop_bounds[t][3]
            corrected_img[t,0,0,:,:,0] = ( #subtract new flatfield from raw mCitr image in the aligned crop boundaries
                temp_raw_mCit[temp_top:temp_bottom,temp_left:temp_right]
                - temp_ff[temp_top:temp_bottom,temp_left:temp_right]
                )
            cropped_raw_img[t,0,0,:,:,0] = ( #subtract new flatfield from raw mCitr image in the aligned crop boundaries
                temp_raw_mCit[temp_top:temp_bottom,temp_left:temp_right]
                )
            corrected_img[corrected_img < 0] = 0 # no negative numbers
            
        print('finished making corrected images for position' + temp_posname)

        #corrects the metatadata
        metadata_df = pd.read_csv(path.join(pos_dir,file_prefix+'metadata.csv')) #read metadata
        metadata_df['values'][metadata_df['Description']=='channel_1_name'] = ch_name + 'FFC'
        channel_dict = { #establish channel name and wavelength to append to metadata
            'Description':['channel_3_name', 'channel_3_emWavelen','channel_4_name', 'channel_4_emWavelen'],
            'values':[ch_name + 'oldffc', 0.0,ch_name + 'Raw', 0.0]
           }
        channel_df = pd.DataFrame(channel_dict) #turn this into dataframe to concatenate later
        
        metadata_output = pd.concat([metadata_df,channel_df],ignore_index = True) #append new rows to dataframe
        metadata_output.to_csv(path.join(pos_dir,file_prefix+'metadata.csv'), index=False) #save metadata
        
        #renames old ffc files
        os.rename(path.join(pos_dir,file_prefix+ch_name+'.tif'), 
                  path.join(pos_dir,file_prefix+ch_name+'oldffc.tif'))
        os.rename(path.join(pos_dir,file_prefix+ch_name+'_aligned.npz'), 
                  path.join(pos_dir,file_prefix+ch_name+'oldffc_aligned.npz'))

        #writes new ffc files
        tifffile.imwrite(path.join(pos_dir,file_prefix+ch_name+'FFC.tif'),
                         corrected_img, imagej=True, metadata={'axes': 'TZCYXS'})
        np.savez(path.join(pos_dir,file_prefix+ch_name+'FFC_aligned.npz'), np.squeeze(corrected_img))
        tifffile.imwrite(path.join(pos_dir,file_prefix+ch_name+'Raw.tif'),
                         cropped_raw_img, imagej=True, metadata={'axes': 'TZCYXS'})
        np.savez(path.join(pos_dir,file_prefix+ch_name+'Raw_aligned.npz'), np.squeeze(cropped_raw_img))
        print('saved position' + temp_posname)
    print('finished_'+expt_names[n])

    