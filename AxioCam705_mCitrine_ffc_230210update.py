# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:43:01 2023

@author: Jacob

This is an updated version of the script that converts a single CZI file with 
many positions into one multichannel tif per position, while also performing 
a darkfield subtraction and flatfield division on the second (yellow) channel.
 
Output TIF dimensions: [num_frames] time frames, 3 channels (Phase, YFP, RFP).

Updated: 2022/07/12
"""

"""
JX NOTE: THE ABOVE DOCUMENTATION IS OUTDATED. NEED TO REVISE THIS.
The output is the original image file with two additional channels concatenated:
    n+1: regular flatfield subtraction
    n+2: scaled flatfield correction

"""

# In[]:

# NOTE: make sure aicsimageio, aicspylibczi, tiffile, etc. are installed in ACDC Python env (pip install ...)
from aicsimageio import AICSImage 
from aicspylibczi import CziFile
import numpy as np
import tifffile
from os import path

# In[]:

# INPUT
img_file = r"E:\DATA\JK FKH\240321_JX_JK137_0aTc\New-01.czi" # path to CZI file to be flatfield-corrected

# OUTPUT
output_dir = r"E:\DATA\JK FKH\240321_JX_JK137_0aTc" # directory to save flatfield-corrected tif
output_name = "240321_JX_JK137_0aTc" # file name for flatfield-corrected mCitrine tif

# FLATFIELD
ff_file = r"C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\Flat and dark fields\\230210_FFcombined_mean.TIF" # path to flatfield image
df_file = r"C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\Flat and dark fields\\230210_DFcombined_mean.TIF"
net_ff_file = r"C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\Flat and dark fields\\230210_scaled_net_flatfield.npy"

# FF explanations:
# ff is mean of multiple stills with no cells, yellow illumination (currently only used for the old method)
# df is mean of multiple stills with no cells, no illumination
# net ff is ff minus df, divided by the mean of all pixels (so average pixel value is 1)

num_frames = 121



# In[]:

img_czi = CziFile(img_file) # load CZI file
ff_img = AICSImage(ff_file) # load flatfield image
df_img = AICSImage(df_file)
with open(net_ff_file, 'rb') as f:
    net_ff_img = np.load(f)

# the following lines reshape the flatfield image into the dimensions needed to subtract properly from the loaded CZI file
ff = np.empty((1,num_frames,1,1028,1216),dtype = 'float')
df = np.empty((1,num_frames,1,1028,1216),dtype = 'float')
temp_ff = ff_img.data
temp_ff = temp_ff.astype(float)
temp_df = df_img.data
temp_df = temp_df.astype(float)
for t in range(num_frames):
    ff[0,t,0,:,:] = temp_ff
    df[0,t,0,:,:] = temp_df
pos = img_czi.size[0]
print('Total positions: ' + str(pos))


# In[]:

# ONLY USE THIS CELL IF SOMEHOW CAN'T READ NUMBER OF POSITIONS (happened to Jordan at least once)
#pos = 18

# In[]:

for p in range(pos):
# for p in range(15,24): # for partial loop
    print('Position ' + str(p+1) + ' started...')
    
    # old FF subtractive method
    temp_img,temp_shp = img_czi.read_image(S=p)
    temp_img = temp_img.astype(float)
    subtracted_img = np.subtract(temp_img[0,:,1,:,:],ff[0,:,0,:,:])
    subtracted_img[subtracted_img < 0] = 0
    subtracted_img = np.expand_dims(subtracted_img,axis = (0,2))
    
    # new method: subtract darkfield, rescale pixel-by-pixel
    temp_df_sub = np.subtract(temp_img[0,:,1,:,:],df[0,:,0,:,:])
    temp_df_sub[temp_df_sub < 0] = 0
    dfc_ffc_img = temp_df_sub/net_ff_img
    dfc_ffc_img = np.expand_dims(dfc_ffc_img,axis = (0,2))
    
    # combine all images
    output_img = np.concatenate((temp_img, subtracted_img, dfc_ffc_img),axis = 2)
    output_img = output_img.astype("uint16")
    temp_path = path.join(output_dir, output_name+"_pos"+str(p+1).zfill(2)+".tif")
    tifffile.imwrite(
    temp_path, output_img[0, :,:,:,:],
    imagej=True,resolution=(1./0.1095238, 1./0.1095238), 
    #imagej=True,resolution=(1./0.16428454082, 1./0.16428454082),
    metadata={'axes': 'TCYX','unit': 'um'}
    )

    print('Position ' + str(p+1) + ' saved!')

print('Done!')



