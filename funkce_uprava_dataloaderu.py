# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 18:27:34 2022

@author: nohel
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 

import glob
from skimage.io import imread
from skimage.color import rgb2gray,rgb2hsv,rgb2xyz
from skimage.morphology import binary_erosion, disk
from skimage.filters import gaussian
import torchvision.transforms.functional as TF
from torch.nn import init
import matplotlib.pyplot as plt

## DATALOADER 
class DataLoader(torch.utils.data.Dataset):
    def __init__(self,path_to_data="D:\Diploma_thesis_segmentation_disc",sigma=60,size_of_erosion=80,output_image_size=[768,768]):
        self.output_image_size=output_image_size
        self.sigma=sigma
        self.size_of_erosion=size_of_erosion
        self.path=path_to_data
        #Drishti_GS       
        
        self.files_img=glob.glob(self.path+'/Drishti-GS/Images/*.png')
        self.files_mask_disc=glob.glob(self.path+'/Drishti-GS/Disc/expert1/*.png')
        self.files_mask_cup=glob.glob(self.path+'/Drishti-GS\Cup\expert1/*.png')
        self.files_fov=glob.glob(self.path+'/Drishti-GS\FOV/*.png')
        
        #REFUGE_train
        self.files_img=self.files_img+glob.glob(self.path+'/REFUGE/Images/Train/*.png')
        self.files_mask_disc=self.files_mask_disc+glob.glob(self.path+'/REFUGE/Disc/Train/*.png')        
        self.files_mask_cup=self.files_mask_cup+glob.glob(self.path+'/REFUGE/Cup/Train/*.png')
        self.files_fov=self.files_fov+glob.glob(self.path+'/REFUGE/Fov/Train/*.png')
        
        
        self.files_img.sort()
        self.files_mask_disc.sort()
        self.files_mask_cup.sort()
        self.files_fov.sort()
        self.num_of_imgs=len(self.files_img)
        
            
    def __len__(self):
        return self.num_of_imgs
    
    def __getitem__(self,index):
        #Načtení obrázků
        img=imread(self.files_img[index]).astype(np.float32)
        mask_cup=imread(self.files_mask_cup[index])
        mask_disc=imread(self.files_mask_disc[index])
        fov=imread(self.files_fov[index])
        
        center_new=Detection_of_disc(img,fov,self.sigma,self.size_of_erosion)
        output_crop_image, output_mask_disc,output_mask_cup=Crop_image(img,mask_disc,mask_cup,self.output_image_size,center_new)
        
        
        
        ## narvat to do uceni jako batch x sirka x vyska 
        maska_ouput_size=(int(2),self.output_image_size[0],self.output_image_size[1]) # velikost vstupniho obrazu
        mask_spojeny=np.zeros(maska_ouput_size)
        mask_spojeny[0,:,:]=output_mask_disc
        mask_spojeny[1,:,:]=output_mask_cup

        mask_spojeny.astype(np.float32)
        
        img=TF.to_tensor(output_crop_image)
        mask=torch.from_numpy(mask_spojeny)
        
        return img,mask
    
    
    
def Detection_of_disc(image,fov,sigma,size_of_erosion):    
    img=rgb2xyz(image).astype(np.float32)
    img=rgb2gray(img).astype(np.float32)
    #plt.imshow(fov)
    BW=binary_erosion(fov,disk(size_of_erosion))
    #plt.imshow(BW)
    #%%
    vertical_len=BW.shape[0]
    step=round(vertical_len/15);
    BW[0:step,:]=0;
    BW[vertical_len-step:vertical_len,:]=0;
    #plt.imshow(BW)
    
    #%%
    #plt.imshow(img)
    img[~BW]=0;
    #plt.imshow(img)
    #%%
    img_filt=gaussian(img,sigma);
    img_filt[~BW]=0;
    #plt.imshow(img_filt)
    #%%
    max_xy = np.where(img_filt == img_filt.max() )
    r=max_xy[0][0]
    c=max_xy[1][0]
    center_new=[]
    center_new.append(c)
    center_new.append(r)
    return center_new
    
def Crop_image(image,mask_disc,mask_cup,output_image_size,center_new): 
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
    output_mask_disc=mask_disc[x_start:x_start+output_image_size[0],y_start:y_start+output_image_size[1]]
    output_mask_cup=mask_cup[x_start:x_start+output_image_size[0],y_start:y_start+output_image_size[1]]
    return output_crop_image, output_mask_disc,output_mask_cup