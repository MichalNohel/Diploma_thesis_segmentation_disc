# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 18:42:42 2022

@author: nohel
"""

from funkce_final import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np 
from skimage.color import hsv2rgb

if __name__ == "__main__":  
    batch=1 
    loader=DataLoader(split="Train",path_to_data="D:\Diploma_thesis_segmentation_disc/Data_500_500",color_preprocesing="HSV",segmentation_type="disc")
    trainloader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=False)
    
    loader=DataLoader(split="Test",path_to_data="D:\Diploma_thesis_segmentation_disc/Data_500_500",color_preprocesing="HSV",segmentation_type="disc")
    testloader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=False)
    
    
    '''
    for it,(data,mask) in enumerate(trainloader): ### you can iterate over dataset (one epoch)
        if it==0:
            plt.figure(figsize=[10,10])
            plt.subplot(1,2,1)
            #plt.imshow(data[0,0,:,:].detach().cpu().numpy(),cmap='gray')
            im_pom=data[0,:,:,:].detach().cpu().numpy()   
            im_pom=np.transpose(im_pom,(1,2,0))
            plt.imshow(im_pom.astype(np.uint8),vmin=0, vmax=255)
            plt.subplot(1,2,2)
            plt.imshow(mask[0,0,:,:].detach().cpu().numpy())
            plt.show()
            break
        '''
            
    '''
        
    for it,(data,mask) in enumerate(trainloader): ### you can iterate over dataset (one epoch)
        if it==0:
            plt.figure(figsize=[10,10])
            plt.subplot(1,2,1)
            #plt.imshow(data[0,0,:,:].detach().cpu().numpy(),cmap='gray')
            im_pom=data[0,:,:,:].detach().cpu().numpy()   
            im_pom=np.transpose(im_pom,(1,2,0))
            im_pom=hsv2rgb(im_pom)
            plt.imshow(im_pom)
            plt.show()
            break
    '''
    
    for it,(data,mask,img_orig,disc_orig,cup_orig,coordinates) in enumerate(testloader): ### you can iterate over dataset (one epoch)
        if it==0:
            plt.figure(figsize=[10,10])
            plt.subplot(1,2,1)
            #plt.imshow(data[0,0,:,:].detach().cpu().numpy(),cmap='gray')
            im_pom=data[0,:,:,:].detach().cpu().numpy()   
            im_pom=np.transpose(im_pom,(1,2,0))
            im_pom=hsv2rgb(im_pom)
            plt.imshow(im_pom)
            plt.show()
            
            
            
            break
        
        
        
        
        
    