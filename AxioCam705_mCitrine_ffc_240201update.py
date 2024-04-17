# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:43:01 2023

@author: Jacob

This is an updated version of the script that converts a single CZI file with 
many positions into one multichannel tif per position, while also performing 
a darkfield subtraction and flatfield division on the second (yellow) channel.
 
Output TIF dimensions: [num_frames] time frames, 5 channels (Phase, YFP-Raw, RFP-Raw, YFP-FFC, YFP-ScaledFFC).

Updated: 2024/02/01 by JX
- update since 2023/02/10 version: changed to loop through timepoints instead of making one big stack of identical images to subtract

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
img_file = r"G:\My Drive\JX_DATA\Whi5-mCitrine expression\temp\New-01.czi" # path to CZI file to be flatfield-corrected

# OUTPUT
output_dir = r"G:\My Drive\JX_DATA\Whi5-mCitrine expression\temp" # directory to save flatfield-corrected tif
output_name = "temp" # file name for flatfield-corrected mCitrine tif

# FLATFIELD
ff_file = r"C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\Flat and dark fields\\230210_FFcombined_mean.TIF" # path to flatfield image
df_file = r"C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\Flat and dark fields\\230210_DFcombined_mean.TIF"
net_ff_file = r"C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\Flat and dark fields\\230210_scaled_net_flatfield.npy"

# FF explanations:
# ff is mean of multiple stills with no cells, yellow illumination (currently only used for the old method)
# df is mean of multiple stills with no cells, no illumination
# net ff is ff minus df, divided by the mean of all pixels (so average pixel value is 1)

num_frames = 241



# In[]:

img_czi = CziFile(img_file) # load CZI file
ff_img = AICSImage(ff_file) # load flatfield image
df_img = AICSImage(df_file) # load darkfield image
with open(net_ff_file, 'rb') as f:
    net_ff_img = np.load(f)

# the following lines reshape the flatfield image into the dimensions needed to subtract properly from the loaded CZI file
ff = np.empty((1,1,1,1028,1216),dtype = 'float')
df = np.empty((1,1,1,1028,1216),dtype = 'float')

# =============================================================================
# 
# temp_ff = ff_img.data
# temp_ff = temp_ff.astype(float)
# temp_df = df_img.data
# temp_df = temp_df.astype(float)
# for t in range(num_frames):
#     ff[0,t,0,:,:] = temp_ff
#     df[0,t,0,:,:] = temp_df
#     
# =============================================================================
    
ff = ff_img.data
ff = ff.astype(float)
df = df_img.data
df = df.astype(float)

pos = img_czi.size[0] # number of positions in CZI file
print('Total positions: ' + str(pos))


# In[]:

# ONLY USE THIS CELL IF SOMEHOW CAN'T READ NUMBER OF POSITIONS (happened to Jordan at least once)
#pos = 18

# In[]:
pos = 1 # for testing

for p in range(pos):
#for p in range(13,23):
    print('Position ' + str(p+1) + ' started...')
    
    # old method: subtract flatfield
    temp_img,temp_shp = img_czi.read_image(S=p)
    temp_img = temp_img.astype(float)
    # for t in range(num_frames):
    subtracted_img = np.subtract(temp_img[0,:,1,:,:],ff)
    subtracted_img[subtracted_img < 0] = 0
    subtracted_img = np.expand_dims(subtracted_img,axis = (0,2))
        
        
        
# =============================================================================
#         
#     # new method: subtract darkfield, rescale pixel-by-pixel
#     temp_df_sub = np.subtract(temp_img[0,:,1,:,:],df[0,:,0,:,:])
#     temp_df_sub[temp_df_sub < 0] = 0
#     dfc_ffc_img = temp_df_sub/net_ff_img
#     dfc_ffc_img = np.expand_dims(dfc_ffc_img,axis = (0,2))
#     
#     # combine all images
#     output_img = np.concatenate((temp_img, subtracted_img, dfc_ffc_img),axis = 2)
#     output_img = output_img.astype("uint16")
#     temp_path = path.join(output_dir, output_name+"_pos"+str(p+1).zfill(2)+".tif")
#     tifffile.imwrite(
#     temp_path, output_img[0, :,:,:,:],
#     imagej=True,resolution=(1./0.1095238, 1./0.1095238), 
#     #imagej=True,resolution=(1./0.16428454082, 1./0.16428454082),
#     metadata={'axes': 'TCYX','unit': 'um'}
#     )
# 
#     print('Position ' + str(p+1) + ' saved!')
#     
#     
# =============================================================================
    
    

print('Done!')



