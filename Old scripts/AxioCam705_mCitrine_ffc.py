#!/usr/bin/env python
# coding: utf-8

'''
@author: jmhkim

This script converts a single CZI file with many positions into one multichannel tif per position, while also performing flatfield background correction on the second (yellow) channel.
 
Output TIF dimensions: 121 time frames, 3 channels (Phase, YFP, RFP).

Updated: 2022/07/12
'''


# In[]: IMPORT STATEMENTS

# NOTE: make sure aicsimageio, aicspylibczi, tiffile, etc. are installed in ACDC Python env (pip install ...)
from aicsimageio import AICSImage 
from aicspylibczi import CziFile
import numpy as np
import tifffile

# In[]: USER INPUTS

# INPUT
img_file = r"G:\My Drive\JX_DATA\CLN2pr expression\230211_JX_JX99c\New-01.czi" # path to CZI file to be flatfield-corrected

# OUTPUT
output_dir = r"G:\My Drive\JX_DATA\CLN2pr expression\230211_JX_JX99c/" # directory to save flatfield-corrected tif
output_name = "230211_JX_JX99c" # file name for flatfield-corrected mCitrine tif

# FLATFIELD
ff_file = r"C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\Flat and dark fields\230205_flatfield_median.tif" # path to flatfield image

# MOVIE PARAMETERS
numframes = 121
width = 1216 # pixels
height = 1028 # pixels

# In[]: 

img_czi = CziFile(img_file) # load CZI file
ff_img = AICSImage(ff_file) # load flatfield image

# the following lines reshape the flatfield image into the dimensions needed to subtract properly from the loaded CZI file
ff = np.empty((1,numframes,1,height,width),dtype = 'float')
temp_ff = ff_img.data
temp_ff = temp_ff.astype(float)
for t in range(numframes):
    ff[0,t,0,:,:] = temp_ff
    
pos = img_czi.size[0]
print('Total positions: ' + str(pos))

# =============================================================================
# # In[]:
# 
# # ONLY USE THIS IF SOMEHOW CAN'T READ NUMBER OF POSITIONS (happened to Jordan at least once)
# #pos = 18
# 
# p = 22
# print('Position ' + str(p+1) + ' started...')
# 
# temp_img,temp_shp = img_czi.read_image(S=p)
# temp_img = temp_img.astype(float)
# subtracted_img = np.subtract(temp_img[0,:,1,:,:],ff[0,:,0,:,:])
# subtracted_img[subtracted_img < 0] = 0
# subtracted_img = np.expand_dims(subtracted_img,axis = (0,2))
# output_img = np.concatenate((temp_img, subtracted_img),axis = 2)
# output_img = output_img.astype("uint16")
# tifffile.imwrite(
# output_dir + output_name+"_pos"+str(p+1).zfill(2)+".tif", output_img[0,:,:,:,:],
# imagej=True,resolution=(1./0.1095238, 1./0.1095238), metadata={'axes': 'TCYX','unit': 'um'})
# 
# print('Position ' + str(p+1) + ' saved!')
# =============================================================================

# In[]:

for p in range(pos):
    
    print('Position ' + str(p+1) + ' started...')
    
    temp_img,temp_shp = img_czi.read_image(S=p)
    temp_img = temp_img.astype(float)
    subtracted_img = np.subtract(temp_img[0,:,1,:,:],ff[0,:,0,:,:])
    subtracted_img[subtracted_img < 0] = 0
    subtracted_img = np.expand_dims(subtracted_img,axis = (0,2))
    output_img = np.concatenate((temp_img, subtracted_img),axis = 2)
    output_img = output_img.astype("uint16")
    tifffile.imwrite(
    output_dir + output_name+"_pos"+str(p+1).zfill(2)+".tif", output_img[0,:,:,:,:],
    imagej=True,resolution=(1./0.1095238, 1./0.1095238), metadata={'axes': 'TCYX','unit': 'um'})

    print('Position ' + str(p+1) + ' saved!')

print('Done!')


