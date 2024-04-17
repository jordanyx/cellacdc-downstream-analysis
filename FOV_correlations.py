# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:18:18 2023

FIELD OF VIEW CORRELATIONS

@author: jyxiao
"""

# NOTE: make sure aicsimageio, aicspylibczi, tiffile, etc. are installed in ACDC Python env (pip install ...)
from PIL import Image
import numpy as np
from scipy.stats import pearsonr

import seaborn as sns
import matplotlib.pyplot as plt

# used to rebin images
def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


# In[]: LOAD IMAGES

mainpath = r'C:\Users\jyxiao\Documents\GitHub\skotheim-cellacdc-analysis\Flat and dark fields\\'

im2_name = r'230210_DFcombined_median.TIF'
im1_name = r'230210_DFcombined_mean.TIF'

# load image and 
im1 = np.array(Image.open(mainpath+im1_name))
im2 = np.array(Image.open(mainpath+im2_name))

# rebin if desired
row_bin_size = 1
col_bin_size = 1

rows = 1028//row_bin_size
cols = 1216//col_bin_size

im1 = rebin(im1, [rows,cols])
im2 = rebin(im2, [rows,cols])

# In[]: FITTING AND PLOTTING

# turn 2D images into 1D arrays for cross-image pixel correlations
x = im1.flatten()
y = im2.flatten()

# linear fit between flattened image arrays
m , b = np.polyfit(x,y,1)
y_pred = m * x + b
corr, _ = pearsonr(x,y)

sns.set(rc={'figure.figsize':(10,8)})
sns.set(font_scale = 2)
FIG = plt.subplot()

plt.scatter(x,y,marker='x') # plot values
plt.plot(x,y_pred,'r') # plot linear fit

bounds = [min(x)*0.7,max(x)*1.5]
plt.plot(bounds,bounds,'k') # plot x=y line
FIG.axes.set_xlim(bounds)
FIG.axes.set_ylim(bounds)

FIG.axes.set_xlabel(im1_name)
FIG.axes.set_ylabel(im2_name)
FIG.axes.set_title('Pixel-pixel correlation, Pearson R = ' + str(round(corr,5)) \
                   + ', ' + str(row_bin_size) + 'x'  + str(col_bin_size) + ' binning')


# In[]: looking at variation over one image

image = im2 - im1
im_name = im2_name + ' minus DF, normed'
ylims = [0.9,1.1]

x_ave = np.mean(image,0)
y_ave = np.mean(image,1)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,8))

fig.suptitle(im_name + ', ' + str(row_bin_size) + 'x'  + str(col_bin_size) + ' binning')

# ax1.plot(range(image.shape[1]),x_ave)
ax1.plot(range(image.shape[1]),x_ave/np.mean(x_ave))
ax1.set_xlabel('Column (0 = left)')
ax1.set_ylabel('Fluo (au)')
ax1.set_xlim([0,image.shape[1]])
ax1.set_ylim(ylims)

# ax2.plot(range(image.shape[0]),y_ave)
ax2.plot(range(image.shape[0]),y_ave/np.mean(y_ave))
ax2.set_xlabel('Row (0 = top)')
ax2.set_ylabel('Fluo (au)')
ax2.set_xlim([0,image.shape[0]])
ax2.set_ylim(ylims)

"""
other things to try:
    - heat map over 2D space (set min value to 0, max value to 1)
    - heat map of the difference between images
"""

# In[]: heat map

# create the x and y coordinate arrays
xx, yy = np.mgrid[0:image.shape[0], 0:image.shape[1]]

# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, image ,rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)

# show it
plt.show()

