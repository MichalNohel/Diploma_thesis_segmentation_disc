# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 18:42:42 2022

@author: nohel
"""

from funkce_final import DataLoader,Unet
import torch
import matplotlib.pyplot as plt
import numpy as np 
from skimage.color import hsv2rgb,xyz2rgb


    

if __name__ == "__main__":  
    batch=1 
    threshold=0.5
    threshold_patch=0.5
    color_preprocesing="RGB"
    segmentation_type="disc"
    
    loader=DataLoader(split="Train",path_to_data="D:\Diploma_thesis_segmentation_disc/Data_500_500",color_preprocesing=color_preprocesing,segmentation_type=segmentation_type)
    trainloader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=False)
    
    loader=DataLoader(split="Test",path_to_data="D:\Diploma_thesis_segmentation_disc/Data_500_500",color_preprocesing=color_preprocesing,segmentation_type=segmentation_type)
    testloader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=False)
    
    loader=DataLoader(split="HRF",path_to_data="D:\Diploma_thesis_segmentation_disc",color_preprocesing=color_preprocesing,segmentation_type=segmentation_type)
    HRF_loader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=False)
    
    net = Unet().cuda()   
    net.load_state_dict(torch.load("model_02.pth"))
    net.eval()
    
    
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
 
                
                
                
    for it,(data,img_orig,coordinates) in enumerate(HRF_loader): ### you can iterate over dataset (one epoch)
        if it<50:
            '''
            plt.figure(figsize=[10,10])
            plt.subplot(1,2,1)
            #plt.imshow(data[0,0,:,:].detach().cpu().numpy(),cmap='gray')
            im_pom=data[0,:,:,:].detach().cpu().numpy()   
            im_pom=np.transpose(im_pom,(1,2,0))
            im_pom=hsv2rgb(im_pom)
            plt.imshow(im_pom)
            plt.show()
            '''
            pom_sourad=coordinates.detach().cpu().numpy()[0]                  
            pom=np.zeros([data.shape[2],data.shape[3]])
            output_mask_all=np.zeros([data.shape[2],data.shape[3]])
            for i in [0,152]:
                for j in [0,152]:
                    pom_data=data[:,:,i:i+448,j:j+448]
                    pom_data=pom_data.cuda()
                    
                    #vypocet siti
                    output=net(pom_data)
                    output=torch.sigmoid(output)
                    output_mask=output.detach().cpu().numpy() > threshold
                    output_mask_all[i:i+448,j:j+448]=output_mask_all[i:i+448,j:j+448]+output_mask
                    pom[i:i+448,j:j+448]=pom[i:i+448,j:j+448]+1
                    #plt.imshow(output_mask_all)
            output_mask_all=output_mask_all/pom
            output_mask_all=output_mask_all>threshold_patch
            
            output=np.zeros([img_orig.shape[1],img_orig.shape[2]])
            
            
            if (pom_sourad[1]-300<0):
                x_start=0
            elif((pom_sourad[1]+300)>output.shape[0]):
                x_start=output.shape[0]-600
            else:
                x_start=pom_sourad[1]-300
                
            if (pom_sourad[0]-300<0):
                y_start=0
            elif((pom_sourad[0]+300)>output.shape[1]):
                y_start=output.shape[1]-600
            else:
                y_start=pom_sourad[0]-300
                

            output[x_start:x_start+600,y_start:y_start+600]=output_mask_all
            output_mask=output.astype(bool)
            
            plt.figure(figsize=[10,10])
            plt.subplot(2,2,1)        
            im_pom=img_orig[0,:,:,:].detach().cpu().numpy()   
            plt.imshow(im_pom)        
            
            plt.subplot(2,2,2)    
            plt.imshow(output_mask)
            
            plt.subplot(2,2,3)        
            data_pom=data[0,:,:,:].detach().cpu().numpy()  
            data_pom=np.transpose(data_pom,(1,2,0))
            #data_pom=xyz2rgb(data_pom)
            plt.imshow(data_pom.astype(np.uint8))  
            
            plt.subplot(2,2,4)                       
            plt.imshow(output_mask_all)            
            plt.show()
                        
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    