# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:49:43 2022

@author: nohel
"""

import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray,rgb2hsv,rgb2xyz
from skimage.morphology import binary_erosion, disk
from skimage.filters import gaussian
import matplotlib.pyplot as plt


if __name__ == "__main__": 
    image=imread('D:\Diploma_thesis_segmentation_disc\Drishti-GS\Images\drishti_test_na_001.png')
    fov=imread('D:\Diploma_thesis_segmentation_disc\Drishti-GS\FOV\drishti_test_na_001_fov.png').astype(bool)
    sigma=60
    size_of_erosion=80
    # %%
    
    
    img=rgb2xyz(image).astype(np.float32)
    img=rgb2gray(img).astype(np.float32)
    plt.imshow(fov)
    BW=binary_erosion(fov,disk(size_of_erosion))
    plt.imshow(BW)
    #%%
    vertical_len=BW.shape[0]
    step=round(vertical_len/15);
    BW[0:step,:]=0;
    BW[vertical_len-step:vertical_len,:]=0;
    plt.imshow(BW)
    
    #%%
    plt.imshow(img)
    img[~BW]=0;
    plt.imshow(img)
    #%%
    img_filt=gaussian(img,sigma);
    img_filt[~BW]=0;
    plt.imshow(img_filt)
    #%%
    max_xy = np.where(img_filt == img_filt.max() )
    r=max_xy[0][0]
    c=max_xy[1][0]
    center_new=[]
    center_new.append(c)
    center_new.append(r)
    
    
    
    

