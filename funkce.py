# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 12:18:20 2021

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
    def __init__(self,split="Train",path_to_data="REFUGE_VYGENEROVANE_PATCHE_256_64", predzpracovani="sedoton"):
        self.split=split
        self.path=path_to_data+'/'+split
        self.predzpracovani=predzpracovani
        
        if split=="Train":
            self.files_img=glob.glob(self.path+'/REFUGE_Training_400_patche/*.png')
            self.files_mask=glob.glob(self.path+'/REFUGE_Training_400_patche_GT/*.png')
            self.files_img.sort()
            self.files_mask.sort()
            self.num_of_imgs=len(self.files_img)
            
        if split=="Validation":
            self.files_img=glob.glob(self.path+'/REFUGE_Validation_400_patche/*.png')
            self.files_mask=glob.glob(self.path+'/REFUGE_Validation_400_patche_GT/*.png')
            self.files_img.sort()
            self.files_mask.sort()
            self.num_of_imgs=len(self.files_img)
        
        if split=="Test":
            self.files_img=glob.glob(self.path+'/REFUGE_Test_400_patche/*.png')
            self.files_mask=glob.glob(self.path+'/REFUGE_Test_400_patche_GT/*.png')
            self.files_img.sort()
            self.files_mask.sort()
            self.num_of_imgs=len(self.files_img)
            
    def __len__(self):
        return self.num_of_imgs
    
    def __getitem__(self,index):
        #Načtení obrázků
        img=imread(self.files_img[index])
        mask=imread(self.files_mask[index])
        
        #Preprocesing
        if(self.predzpracovani=="sedoton"):
            img=rgb2gray(img).astype(np.float32)
            
        if(self.predzpracovani=="barevne"):
            img=img.astype(np.float32)
            
        if(self.predzpracovani=="HSV"):
            img=rgb2hsv(img).astype(np.float32)
            
        if(self.predzpracovani=="XYZ"):
            img=rgb2xyz(img).astype(np.float32)
            
        mask=mask.astype(np.float32)
        
        #rozdělení na cup a disk
        
        mask_cup=np.zeros(mask.shape)
        mask_disk=np.zeros(mask.shape)
        
        mask_cup[mask==0]=1
        mask_disk[mask!=255]=1

        mask_cup.astype(np.float32)
        mask_disk.astype(np.float32)   
        
        
        ## narvat to do uceni jako batch x sirka x vyska 
        maska_ouput_size=(int(2),int(256),int(256)) # velikost vstupniho obrazu
        mask_spojeny=np.zeros(maska_ouput_size)
        mask_spojeny[0,:,:]=mask_disk
        mask_spojeny[1,:,:]=mask_cup
        
        #maska_ouput_size=(int(1),int(256),int(256)) # velikost vstupniho obrazu
        #mask_spojeny=np.zeros(maska_ouput_size)
        #mask_spojeny[0,:,:]=mask_disk
        #mask_spojeny[0,:,:]=mask_cup
        
        mask_spojeny.astype(np.float32)
        
        img=TF.to_tensor(img)
        mask=torch.from_numpy(mask_spojeny)
        
        return img,mask
    
        
## U-Net

class unetConv2(nn.Module):
    def __init__(self,in_size,out_size,filter_size=3,stride=1,pad=1,do_batch=1):
        super().__init__()
        self.do_batch=do_batch
        self.conv=nn.Conv2d(in_size,out_size,filter_size,stride,pad)
        self.bn=nn.BatchNorm2d(out_size,momentum=0.1)
        
    def forward(self,inputs):
        outputs=self.conv(inputs)    
        
        if self.do_batch:
            outputs=self.bn(outputs) 
        
        outputs=F.relu(outputs)
        return outputs
        
class unetConvT2(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=2,pad=1,out_pad=1):
        super().__init__()
        self.conv=nn.ConvTranspose2d(in_size, out_size, filter_size,stride=stride, padding=pad, output_padding=out_pad)
        
    def forward(self,inputs):
        outputs=self.conv(inputs)
        outputs=F.relu(outputs)
        return outputs

        
class unetUp(nn.Module):
    def __init__(self,in_size,out_size):
        super(unetUp,self).__init__()
        self.up = unetConvT2(in_size,out_size)
        
    def forward(self,inputs1, inputs2):
        inputs2=self.up(inputs2)
        return torch.cat([inputs1,inputs2],1)
    
class Unet(nn.Module):
    def __init__(self, filters=(np.array([16, 32, 64, 128, 256])/2).astype(np.int32),in_size=3,out_size=1):
        super().__init__()
        self.out_size=out_size
        self.in_size=in_size
        self.filters=filters
        
        self.conv1 = nn.Sequential(unetConv2(in_size, filters[0]),unetConv2(filters[0], filters[0]),unetConv2(filters[0], filters[0]))
    
        self.conv2 = nn.Sequential(unetConv2(filters[0], filters[1] ),unetConv2(filters[1], filters[1] ),unetConv2(filters[1], filters[1] ))
        
        self.conv3 = nn.Sequential(unetConv2(filters[1], filters[2] ),unetConv2(filters[2], filters[2] ),unetConv2(filters[2], filters[2] ))

        self.conv4 = nn.Sequential(unetConv2(filters[2], filters[3] ),unetConv2(filters[3], filters[3] ),unetConv2(filters[3], filters[3] ))



        self.center = nn.Sequential(unetConv2(filters[-2], filters[-1] ),unetConv2(filters[-1], filters[-1] )) 
          
        
        
        self.up_concat4 = unetUp(filters[4], filters[4] )        
        self.up_conv4=nn.Sequential(unetConv2(filters[3]+filters[4], filters[3] ),unetConv2(filters[3], filters[3] ))

        self.up_concat3 = unetUp(filters[3], filters[3] )
        self.up_conv3=nn.Sequential(unetConv2(filters[2]+filters[3], filters[2] ),unetConv2(filters[2], filters[2] ))

        self.up_concat2 = unetUp(filters[2], filters[2] )
        self.up_conv2=nn.Sequential(unetConv2(filters[1]+filters[2], filters[1] ),unetConv2(filters[1], filters[1] ))
    
        self.up_concat1 = unetUp(filters[1], filters[1] )
        self.up_conv1=nn.Sequential(unetConv2(filters[0]+filters[1], filters[0] ),unetConv2(filters[0], filters[0],do_batch=0 ))
            
        self.final = nn.Conv2d(filters[0], self.out_size, 1)
        
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
    
    
    def forward(self,inputs):
        conv1=self.conv1(inputs)
        x=F.max_pool2d(conv1,2,2)
        
        conv2=self.conv2(x)
        x=F.max_pool2d(conv2,2,2)
        
        conv3=self.conv3(x)
        x=F.max_pool2d(conv3,2,2)
        
        conv4=self.conv4(x)
        x=F.max_pool2d(conv4,2,2)
        
        x=self.center(x)
        
        x=self.up_concat4(conv4,x)
        x=self.up_conv4(x)
        
        x=self.up_concat3(conv3,x)
        x=self.up_conv3(x)
        
        x=self.up_concat2(conv2,x)
        x=self.up_conv2(x)
        
        x=self.up_concat1(conv1,x)
        x=self.up_conv1(x)
    
        x=self.final(x)
        
        return x
    

def dice_loss(X,Y):
    eps=1.
    dice=((2. * torch.sum(X*Y) + eps) / (torch.sum(X) + torch.sum(Y) + eps) )
    return 1-dice



def rekonstrukce_masky_z_patchu(dataset_output,dataset_lbl,prah):
    pocet_kroku=13
    velikost_patche=256
    posun=64
    pom=np.zeros([1024,1024])
    rekonstrukce_lbl=np.zeros([1024,1024])
    k=0
    for i in range(pocet_kroku):
        for j in range(pocet_kroku):
            pom_output=dataset_output[k]
            pom_mask=dataset_lbl[k]            
            pom[i*posun:i*posun+velikost_patche,j*posun:j*posun+velikost_patche]=pom[i*posun:i*posun+velikost_patche,j*posun:j*posun+velikost_patche]+pom_output
            rekonstrukce_lbl[i*posun:i*posun+velikost_patche,j*posun:j*posun+velikost_patche]=rekonstrukce_lbl[i*posun:i*posun+velikost_patche,j*posun:j*posun+velikost_patche]+pom_mask
            k=k+1
            
            '''
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(rekonstrukce_lbl)
            plt.subplot(1,2,2)
            plt.imshow(rekonstrukce_output)
            plt.show()
            '''
    rekonstrukce_output=np.zeros([1024,1024])
    rekonstrukce_output[pom>prah]=1
    rekonstrukce_lbl[rekonstrukce_lbl>0]=1
        
    return rekonstrukce_output, rekonstrukce_lbl
            
def dice_coefficient(X,Y):
    # X-otuput, Y-Label
    TP=np.sum(np.logical_and(X,Y))
    FP=np.sum(np.logical_and(np.logical_not(X),Y))
    FN=np.sum(np.logical_and(X,np.logical_not(Y)))
    dice = 2*TP/(2*TP+FP+FN)
    return dice

def Sensitivity (X,Y):
    # X-otuput, Y-Label
    TP=np.sum(np.logical_and(X,Y))
    FN=np.sum(np.logical_and(X,np.logical_not(Y)))
    sensitivity = TP/(FN+TP)
    return sensitivity
    
def Specificity (X,Y):
    # X-otuput, Y-Label
    TN=np.sum(np.logical_and(np.logical_not(X),np.logical_not(Y)))
    FP=np.sum(np.logical_and(np.logical_not(X),Y))
    specificity = TN/(TN+FP)
    return specificity

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
    
def Crop_image(image,output_image_size,center_new): 
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
    return output_crop_image
    



