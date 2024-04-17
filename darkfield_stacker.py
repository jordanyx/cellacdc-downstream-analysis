# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:52:00 2023
@author: Jacob
"""
'''
This script takes in darkfield images from their own separate czi files and
outputs a single TIFF file with all the images in a hyperstack, and a TIF file
of the median pixel values.
'''
#%%
# NOTE: make sure aicsimageio, aicspylibczi, tiffile, etc. are installed in ACDC Python env (pip install ...)
from aicspylibczi import CziFile
import numpy as np
import tifffile as tf
from os import path
from glob import glob

#%%
darkfield_folder = r"G:\My Drive\JX_DATA\Darkfield CZIs\Alltogether\\"
darkfield_dirs = glob(path.join(darkfield_folder,'*.czi'))

#%% grab dimensions
dim_czi = CziFile(darkfield_dirs[0])
dim_y = dim_czi.size[1]
dim_x = dim_czi.size[2]
dim_t = len(darkfield_dirs)

#%% create empty arrays for the concatenated and median image outputs
conc_img = np.empty((1,dim_t,1,dim_y,dim_x),dtype = 'float')
med_img = np.empty((1,1,1,dim_y,dim_x),dtype = 'float')

#%% create concatenated image
for t in range(dim_t):
    temp_czi = CziFile(darkfield_dirs[t])
    temp_im,temp_shp = temp_czi.read_image()
    conc_img[0,t,0,:,:] = temp_im
uint16_conc_img = conc_img.astype("uint16")

#%% export and save concatenated image
output_dir = r'C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\Flat and dark fields\\'
file_name = r'230210_DFcombined_stack.TIF'
tf.imwrite(
output_dir + file_name,
uint16_conc_img[0, :,:,:,:],
imagej=True,
resolution=(1./0.1095238, 1./0.1095238),
metadata={'axes': 'TCYX','unit': 'um'}
)

#%% create median image
float_median_img = np.median(conc_img,axis=[1])
int_median_img = float_median_img.astype("uint16") #turn float into int

#%% export and save median image
output_dir = r'C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\Flat and dark fields\\'
file_name = r'230210_DFcombined_median.TIF'
tf.imwrite(
output_dir + file_name,
int_median_img,
imagej=True,
resolution=(1./0.1095238, 1./0.1095238),
metadata={'axes': 'TCYX','unit': 'um'}
)

#%% create mean image
float_mean_img = np.mean(conc_img,axis=1)
int_mean_img = float_mean_img.astype("uint16") #turn float into int

#%% export and save mean image
output_dir = r'C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\Flat and dark fields\\'
file_name = r'230210_DFcombined_mean.TIF'
tf.imwrite(
output_dir + file_name,
int_mean_img,
imagej=True,
resolution=(1./0.1095238, 1./0.1095238),
metadata={'axes': 'TCYX','unit': 'um'}
)