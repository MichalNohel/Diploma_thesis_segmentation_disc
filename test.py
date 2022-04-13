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
from funkce import Detection_of_disc, Crop_image

if __name__ == "__main__": 
    image=imread('D:\Diploma_thesis_segmentation_disc\Drishti-GS\Images\drishti_test_na_001.png')
    fov=imread('D:\Diploma_thesis_segmentation_disc\Drishti-GS\FOV\drishti_test_na_001_fov.png').astype(bool)
    mask_disc=imread('D:\Diploma_thesis_segmentation_disc\Drishti-GS\Disc\expert1\drishti_test_na_001_disc_exp_1.png')
    mask_cup=imread('D:\Diploma_thesis_segmentation_disc\Drishti-GS\Cup\expert1\drishti_test_na_001_cup_exp_1.png')
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
    
    #%%
    center_new=[]
    center_new=Detection_of_disc(image,fov,sigma,size_of_erosion)
    
    #%%
    plt.figure()
    plt.imshow(image)
    plt.stem(center_new[0],center_new[1])
    plt.show()
    
    #%% Crop image with disc
    
    output_image_size=[768,768]
    size_in_img=image.shape
    x_half=int(output_image_size[0]/2)
    y_half=int(output_image_size[1]/2)
    
    
    if ((center_new[1]-x_half)<0):
        x_start=0
    elif ((center_new[1]+x_half)>size_in_img[0]):
        x_start=size_in_img[0]-output_image_size[0]
    else:
        x_start=center_new[1]-x_half
            
        
    
    if ((center_new[0]-y_half)<0):
        y_start=0
    elif ((center_new[0]+y_half)>size_in_img[1]):
        y_start=size_in_img[1]-output_image_size[1]
    else:
        y_start=center_new[0]-y_half
    
    output_crop_image=image[x_start:x_start+output_image_size[0],y_start:y_start+output_image_size[1],:]
    
    plt.figure()
    plt.imshow(output_crop_image)
    plt.show()
    
    #%%
    output_image_size=[768,768]
    output_crop_image, output_mask_disc,output_mask_cup=Crop_image(image,mask_disc,mask_cup,output_image_size,center_new)
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.imshow(output_crop_image)
    plt.subplot(3,1,2)
    plt.imshow(output_mask_disc)
    plt.subplot(3,1,3)
    plt.imshow(output_mask_cup)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

