# -*- coding: utf-8 -*-
"""
Notes:

The mCitrine aligned npz file only contains the images as an ndarray. You can access
the array by using data[data.files[0]], where data is the output from np.load

The background ROI data npz file contains the background region image as an array.
If we want, we can implement a system to include a new region like this in the new 
flatfield image.

the align_shift.mpy file contains a nested array. There is an array for each 
timepoint, and the nested array just has 2 values: the horz and vert pixel shift.
I do not yet know which value corresponds to which direction. It will have to be
figured out through experimentation

1) Apply the background subtraction
    - Modify the existing code
2) Shift the pixels.
    - Move the image in one direction while simultaneously filling in the empties
      with zeros and deleting the pixels that move outside the boundaries.
3) Crop the image
    - Use the bounding dataprepROI_coords csv file
    - Need to figure out if the values in the coordinates are inclusive or not
4) Save the image as both a TIF and an npz in the same format with the same name
    - Don't forget to renamne the old image
"""
#%%

from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from aicspylibczi import CziFile
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
import csv
import pandas as pd
from matplotlib import pyplot as plt

#%% 

#img_file = r"C:\Users\Jacob\OneDrive\Documents\AxioCam705_control_images\\220415_yJK056_autof.czi"
output_dir = r"G:\My Drive\ffc_test\bulktest\acdc_files\220607_SM_yJK098_30ngmLaTc_expt3_data\\"
output_name = r"220607_SM_yJK098_30ngmLaTc_expt3_oldff"
align_coords = np.load(r'D:\Downloads\Images\220607_SM_yJK098_30ngmLaTc_expt3_pos03_s1_align_shift.npy')
crop_bound_file = r"D:\Downloads\Images\220607_SM_yJK098_30ngmLaTc_expt3_pos03_s1_dataPrepROIs_coords.csv"
nframes = len(align_coords)


img_file = r"D:\Downloads\New-03.czi" #czi file from expt
ff_file = r"D:\Downloads\\AxioCam705_mCitrine_ff.TIF" #path of flatfield median image
img_czi = CziFile(img_file)

ff_img = AICSImage(ff_file)
ff = np.empty((1,121,1,1028,1216),dtype = 'float')
temp_ff = ff_img.data
temp_ff = temp_ff[0,0,0,:,:]
temp_ff = temp_ff.astype(float)
for t in range(121):
    ff[0,t,0,:,:] = temp_ff

#%%

print(img_czi.size)
#%% background correction
'''
pos = img_czi.size[0]

#for p in range(pos):
p = 0
temp_img,temp_shp = img_czi.read_image()
temp_img = temp_img.astype(float)
temp_img[p,:,1,:,:] -= ff[0,:,0,:,:]

temp_img[temp_img < 0] = 0
#temp_img = temp_img.astype("uint16")
'''
#%% get new crop coordinates for each frame based on alignment

crop_bounds = pd.read_csv(crop_bound_file)
crop_bounds_values = crop_bounds["value"]
crop_bounds_array = np.array([crop_bounds_values[0],crop_bounds_values[1],
                              crop_bounds_values[2],crop_bounds_values[3]])

aligned_crop_bounds = np.empty(shape=[nframes,4],dtype='int64')

for n in range(nframes):
    temp_align_coords = np.array([align_coords[n,1],align_coords[n,1],
                                  align_coords[n,0],align_coords[n,0]])
    aligned_crop_bounds[n] = crop_bounds_array - temp_align_coords

#%% make subtraction in cropped and aligned rectangle
corrected_img = np.empty(shape=[nframes,
                                1,
                                1,
                                crop_bounds_values[3]-crop_bounds_values[2],
                                crop_bounds_values[1]-crop_bounds_values[0],
                                1],dtype='int16')

for t in range(nframes):
    temp_raw_mCit,_ = img_czi.read_image(S=2, C=1, T=t)
    temp_raw_mCit = np.squeeze(temp_raw_mCit)
    temp_left = aligned_crop_bounds[t][0]
    temp_right = aligned_crop_bounds[t][1]
    temp_top = aligned_crop_bounds[t][2]
    temp_bottom = aligned_crop_bounds[t][3]
    corrected_img[t,0,0,:,:,0] = temp_raw_mCit[temp_top:temp_bottom,temp_left:temp_right] - temp_ff[temp_top:temp_bottom,temp_left:temp_right]
corrected_img[corrected_img < 0] = 0

#%% save corrected image

tifffile.imwrite(output_dir + "test.tif", corrected_img, imagej=True, 
                 metadata={'axes': 'TZCYXS'})
np.savez(output_dir + "test.npz",np.squeeze(corrected_img))

#%%

foo = np.load(output_dir + "test.npz")
#foo = np.load(output_dir + "220607_SM_yJK098_30ngmLaTc_expt3_pos03_s1_mKate2_aligned.npz")
bar = foo[foo.files[0]]
