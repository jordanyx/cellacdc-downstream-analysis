#!/usr/bin/env python
# coding: utf-8

# In[]: IMPORTS

# NOTE: make sure aicsimageio, aicspylibczi, tiffile, etc. are installed in ACDC Python env (pip install ...)

from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from aicspylibczi import CziFile
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile as tf


# This script generates a flatfield image for one fluorescence channel by taking the median for a given pixel, across many single snapshots.
# 
# Output: Single tif
# 
# Updated: 2022/05/03

# In[21]: CHOOSE CZI FILE

ff_file = r"G:\My Drive\JX_DATA\scaling stuff for mike\230915_JX_yellow800ms_flatfield\New-02.czi" 

# ff_file = r"G:\My Drive\JX_DATA\Flatfield CZIs\230119_flatfield.czi" 
# ff_file2 = r"G:\My Drive\JX_DATA\Flatfield CZIs\230203_flatfield.czi" 
# ff_file3 = r"G:\My Drive\JX_DATA\Flatfield CZIs\230205_flatfield.czi" 
# ff_file4 = r"G:\My Drive\JX_DATA\Flatfield CZIs\230208_flatfield.czi" 
# ff_file5 = r"G:\My Drive\JX_DATA\Flatfield CZIs\230209_flatfield.czi" 
#directory of czi file. keep as r-string to avoid errors due to space (Google Drive won't let me change the directory name U+1F620)


# In[]: OLD, BAD
# =============================================================================
# 
# ff_czi = CziFile(ff_file) #load czi file
# median_ff = np.empty((ff_czi.size[2],ff_czi.size[3])) #create empty array of the appropriate dimensions for your final output
# img_ff,shp_ff = ff_czi.read_image(C = 1) #read czi file. img_ff = numpy array, shp_ff = shape of array
# float_median_ff = np.median(img_ff[0],axis=[0]) #get median pixel values from flatfield images
# int_median_ff = float_median_ff.astype(int) #turn float into int
# im = Image.fromarray(int_median_ff) #turn array into image type
# 
# =============================================================================

# In[]: MEDIAN IMAGE

ff_czi = CziFile(ff_file) #load czi file
# ff_czi2 = CziFile(ff_file2) #load czi file
# ff_czi3 = CziFile(ff_file3) #load czi file
# ff_czi4 = CziFile(ff_file4) #load czi file
# ff_czi5 = CziFile(ff_file5) #load czi file

median_ff = np.empty((ff_czi.size[2],ff_czi.size[3])) #create empty array of the appropriate dimensions for your final output

img_ff,shp_ff = ff_czi.read_image(C = 1) #read czi file. img_ff = numpy array, shp_ff = shape of array
# img_ff2,shp_ff2 = ff_czi2.read_image(C = 1)
# img_ff3,shp_ff3 = ff_czi3.read_image(C = 1)
# img_ff4,shp_ff4 = ff_czi4.read_image(C = 1)
# img_ff5,shp_ff5 = ff_czi5.read_image(C = 1)

# In[]

img_ff = np.concatenate((img_ff,), axis=0)
# img_ff = np.concatenate((img_ff,img_ff2,img_ff3,img_ff4,img_ff5), axis=0)

float_median_ff = np.median(img_ff,axis=0) #get median pixel values from flatfield images
int_median_ff = float_median_ff.astype(int) #turn float into int
im = Image.fromarray(int_median_ff[0,:,:]) #turn array into image type

im.save(r"G:\My Drive\JX_DATA\scaling stuff for mike\230915_JX_yellow800ms_flatfield\230915_JX_yellow800ms_flatfield_median.tif") #save the image


# In[ ]: MEAN IMAGE

ff_czi = CziFile(ff_file) #load czi file
mean_ff = np.empty((ff_czi.size[2],ff_czi.size[3])) #create empty array of the appropriate dimensions for your final output
img_ff,shp_ff = ff_czi.read_image(C = 1) #read czi file. img_ff = numpy array, shp_ff = shape of array
float_mean_ff = np.mean(img_ff,axis=0) #get median pixel values from flatfield images
int_mean_ff = float_mean_ff.astype(int) #turn float into int
im = Image.fromarray(int_mean_ff[0,:,:]) #turn array into image type

im.save(r"G:\My Drive\JX_DATA\scaling stuff for mike\230915_JX_yellow800ms_flatfield\230915_JX_yellow800ms_flatfield_mean.tif") #save the image





