# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:18:56 2022

@author: nohel
"""
import glob
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray,rgb2hsv,rgb2xyz
from skimage.morphology import disk
from scipy.ndimage import binary_erosion 
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from funkce import Detection_of_disc, Crop_image
import cv2 


if __name__ == "__main__": 
    sigma=60
    size_of_erosion=80
    path_to_data="D:\Diploma_thesis_segmentation_disc"
    output_image_size=[768,768]
    
    #Drishti_GS       
    
    files_img=glob.glob(path_to_data+'/Drishti-GS/Images/*.png')
    files_mask_disc=glob.glob(path_to_data+'/Drishti-GS/Disc/expert1/*.png')
    files_mask_cup=glob.glob(path_to_data+'/Drishti-GS/Cup/expert1/*.png')
    files_fov=glob.glob(path_to_data+'/Drishti-GS/FOV/*.png')
    path_to_crop_image=path_to_data+'/Drishti-GS/Images_crop/'
    path_to_crop_disc=path_to_data+'/Drishti-GS/Disc_crop/'
    path_to_crop_cup=path_to_data+'/Drishti-GS/Cup_crop/'
    
    for i in range(len(files_img)):
        image=imread(files_img[i])
        fov=imread((files_fov[i])).astype(bool)
        mask_disc=imread(files_mask_disc[i])
        mask_cup=imread(files_mask_cup[i])
        
        center_new=Detection_of_disc(image,fov,sigma,size_of_erosion)
        output_crop_image, output_mask_disc,output_mask_cup=Crop_image(image,mask_disc,mask_cup,output_image_size,center_new)
        
        output_mask_disc=output_mask_disc.astype(bool)
        output_mask_cup=output_mask_cup.astype(bool)
        """
        plt.figure()
        plt.subplot(3,1,1)
        plt.imshow(output_crop_image)
        plt.subplot(3,1,2)
        plt.imshow(output_mask_disc)
        plt.subplot(3,1,3)
        plt.imshow(output_mask_cup)
        plt.show()
        """ 
        plt.imsave(path_to_crop_image + files_img[i][54:],output_crop_image)
        plt.imsave(path_to_crop_disc + files_mask_disc[i][60:],output_mask_disc)
        plt.imsave(path_to_crop_cup + files_mask_cup[i][59:],output_mask_cup)
        
       
    #REFUGE_train      
    
    files_img=glob.glob(path_to_data+'/REFUGE/Images/Train/*.png')
    files_mask_disc=glob.glob(path_to_data+'/REFUGE/Disc/Train/*.png')
    files_mask_cup=glob.glob(path_to_data+'/REFUGE/Cup/Train/*.png')
    files_fov=glob.glob(path_to_data+'/REFUGE/FOV/Train/*.png')
    path_to_crop_image=path_to_data+'/REFUGE/Images_crop/'
    path_to_crop_disc=path_to_data+'/REFUGE/Disc_crop/'
    path_to_crop_cup=path_to_data+'/REFUGE/Cup_crop/'
    
    for i in range(len(files_img)):
        image=imread(files_img[i])
        fov=imread((files_fov[i])).astype(bool)
        mask_disc=imread(files_mask_disc[i])
        mask_cup=imread(files_mask_cup[i])
        
        center_new=Detection_of_disc(image,fov,sigma,size_of_erosion)
        output_crop_image, output_mask_disc,output_mask_cup=Crop_image(image,mask_disc,mask_cup,output_image_size,center_new)
        
        output_mask_disc=output_mask_disc.astype(bool)
        output_mask_cup=output_mask_cup.astype(bool)
        """
        plt.figure()
        plt.subplot(3,1,1)
        plt.imshow(output_crop_image)
        plt.subplot(3,1,2)
        plt.imshow(output_mask_disc)
        plt.subplot(3,1,3)
        plt.imshow(output_mask_cup)
        plt.show()
        """  
        plt.imsave(path_to_crop_image + files_img[i][56:],output_crop_image)
        plt.imsave(path_to_crop_disc + files_mask_disc[i][54:],output_mask_disc)
        plt.imsave(path_to_crop_cup + files_mask_cup[i][53:],output_mask_cup)
      
        
    #REFUGE_validation    
    
    files_img=glob.glob(path_to_data+'/REFUGE/Images/Validation/*.png')
    files_mask_disc=glob.glob(path_to_data+'/REFUGE/Disc/Validation/*.png')
    files_mask_cup=glob.glob(path_to_data+'/REFUGE/Cup/Validation/*.png')
    files_fov=glob.glob(path_to_data+'/REFUGE/FOV/Validation/*.png')
    path_to_crop_image=path_to_data+'/REFUGE/Images_crop/'
    path_to_crop_disc=path_to_data+'/REFUGE/Disc_crop/'
    path_to_crop_cup=path_to_data+'/REFUGE/Cup_crop/'
    
    for i in range(len(files_img)):
        image=imread(files_img[i])
        fov=imread((files_fov[i])).astype(bool)
        mask_disc=imread(files_mask_disc[i])
        mask_cup=imread(files_mask_cup[i])
        
        center_new=Detection_of_disc(image,fov,sigma,size_of_erosion)
        output_crop_image, output_mask_disc,output_mask_cup=Crop_image(image,mask_disc,mask_cup,output_image_size,center_new)
        
        output_mask_disc=output_mask_disc.astype(bool)
        output_mask_cup=output_mask_cup.astype(bool)
        '''
        #plt.figure()
        plt.subplot(3,1,1)
        plt.imshow(output_crop_image)
        plt.subplot(3,1,2)
        plt.imshow(output_mask_disc)
        plt.subplot(3,1,3)
        plt.imshow(output_mask_cup)
        plt.show()
        '''
        plt.imsave(path_to_crop_image + files_img[i][61:],output_crop_image)
        plt.imsave(path_to_crop_disc + files_mask_disc[i][59:],output_mask_disc)
        plt.imsave(path_to_crop_cup + files_mask_cup[i][58:],output_mask_cup)
         
    
    # Riga - Bin Rushed
    pocet_train=0.8
    
    files_img=glob.glob(path_to_data+'/RIGA/Images/BinRushed/*.png')
    files_mask_disc=glob.glob(path_to_data+'/RIGA/Disc/BinRushed/expert1/*.png')
    files_mask_cup=glob.glob(path_to_data+'/RIGA/Cup/BinRushed/expert1/*.png')
    files_fov=glob.glob(path_to_data+'/RIGA/FOV/BinRushed/*.png')
    path_to_crop_image=path_to_data+'/RIGA/Images_crop/'
    path_to_crop_disc=path_to_data+'/RIGA/Disc_crop/'
    path_to_crop_cup=path_to_data+'/RIGA/Cup_crop/'
    
    num_of_img=len(files_img)
    pom=int(num_of_img*pocet_train)
    
    for i in range(len(files_img)):
        
        image=imread(files_img[i])
        fov=imread((files_fov[i])).astype(bool)
        mask_disc=imread(files_mask_disc[i])
        mask_cup=imread(files_mask_cup[i])
        
        if i<pom:
            center_new=Detection_of_disc(image,fov,sigma,size_of_erosion)
            output_crop_image, output_mask_disc,output_mask_cup=Crop_image(image,mask_disc,mask_cup,output_image_size,center_new)    
            output_mask_disc=output_mask_disc.astype(bool)
            output_mask_cup=output_mask_cup.astype(bool)   
            
            plt.imsave(path_to_crop_image + 'Train/' + files_img[i][58:],output_crop_image)
            plt.imsave(path_to_crop_disc + 'Train/' + files_mask_disc[i][64:],output_mask_disc)
            plt.imsave(path_to_crop_cup + 'Train/' + files_mask_cup[i][63:],output_mask_cup)
        else:
            plt.imsave(path_to_crop_image + 'Test/' + files_img[i][58:],image)
            plt.imsave(path_to_crop_disc + 'Test/' + files_mask_disc[i][64:],mask_disc)
            plt.imsave(path_to_crop_cup + 'Test/' + files_mask_cup[i][63:],mask_cup)
            
          
    # Riga - Magrabia
    pocet_train=0.8
    
    files_img=glob.glob(path_to_data+'/RIGA/Images/Magrabia/*.png')
    files_mask_disc=glob.glob(path_to_data+'/RIGA/Disc/Magrabia/expert1/*.png')
    files_mask_cup=glob.glob(path_to_data+'/RIGA/Cup/Magrabia/expert1/*.png')
    files_fov=glob.glob(path_to_data+'/RIGA/FOV/Magrabia/*.png')
    path_to_crop_image=path_to_data+'/RIGA/Images_crop/'
    path_to_crop_disc=path_to_data+'/RIGA/Disc_crop/'
    path_to_crop_cup=path_to_data+'/RIGA/Cup_crop/'
    
    num_of_img=len(files_img)
    pom=int(num_of_img*pocet_train)
    
    for i in range(len(files_img)):
        
        image=imread(files_img[i])
        fov=imread((files_fov[i])).astype(bool)
        mask_disc=imread(files_mask_disc[i])
        mask_cup=imread(files_mask_cup[i])
        
        if i<pom:
            center_new=Detection_of_disc(image,fov,sigma,size_of_erosion)
            output_crop_image, output_mask_disc,output_mask_cup=Crop_image(image,mask_disc,mask_cup,output_image_size,center_new)    
            output_mask_disc=output_mask_disc.astype(bool)
            output_mask_cup=output_mask_cup.astype(bool)   
            
            plt.imsave(path_to_crop_image + 'Train/' + files_img[i][57:],output_crop_image)
            plt.imsave(path_to_crop_disc + 'Train/' + files_mask_disc[i][63:],output_mask_disc)
            plt.imsave(path_to_crop_cup + 'Train/' + files_mask_cup[i][62:],output_mask_cup)
        else:
            plt.imsave(path_to_crop_image + 'Test/' + files_img[i][57:],image)
            plt.imsave(path_to_crop_disc + 'Test/' + files_mask_disc[i][63:],mask_disc)
            plt.imsave(path_to_crop_cup + 'Test/' + files_mask_cup[i][62:],mask_cup)
            
    
    # Riga - MESSIDOR
    pocet_train=0.8
    
    files_img=glob.glob(path_to_data+'/RIGA/Images/MESSIDOR/*.png')
    files_mask_disc=glob.glob(path_to_data+'/RIGA/Disc/MESSIDOR/expert1/*.png')
    files_mask_cup=glob.glob(path_to_data+'/RIGA/Cup/MESSIDOR/expert1/*.png')
    files_fov=glob.glob(path_to_data+'/RIGA/FOV/MESSIDOR/*.png')
    path_to_crop_image=path_to_data+'/RIGA/Images_crop/'
    path_to_crop_disc=path_to_data+'/RIGA/Disc_crop/'
    path_to_crop_cup=path_to_data+'/RIGA/Cup_crop/'
    
    num_of_img=len(files_img)
    pom=int(num_of_img*pocet_train)
    
    for i in range(len(files_img)):
        
        image=imread(files_img[i])
        fov=imread((files_fov[i])).astype(bool)
        mask_disc=imread(files_mask_disc[i])
        mask_cup=imread(files_mask_cup[i])
        
        if i<pom:
            center_new=Detection_of_disc(image,fov,sigma,size_of_erosion)
            output_crop_image, output_mask_disc,output_mask_cup=Crop_image(image,mask_disc,mask_cup,output_image_size,center_new)    
            output_mask_disc=output_mask_disc.astype(bool)
            output_mask_cup=output_mask_cup.astype(bool)   
            
            plt.imsave(path_to_crop_image + 'Train/' + files_img[i][57:],output_crop_image)
            plt.imsave(path_to_crop_disc + 'Train/' + files_mask_disc[i][63:],output_mask_disc)
            plt.imsave(path_to_crop_cup + 'Train/' + files_mask_cup[i][62:],output_mask_cup)
        else:
            plt.imsave(path_to_crop_image + 'Test/' + files_img[i][57:],image)
            plt.imsave(path_to_crop_disc + 'Test/' + files_mask_disc[i][63:],mask_disc)
            plt.imsave(path_to_crop_cup + 'Test/' + files_mask_cup[i][62:],mask_cup)
    
    
    # RIM-ONE - glaucoma
    pocet_train=0.8
    
    files_img=glob.glob(path_to_data+'/RIM-ONE/Images/Glaucoma/*.png')
    files_mask_disc=glob.glob(path_to_data+'/RIM-ONE/Disc/Glaucoma/*.png')
    files_mask_cup=glob.glob(path_to_data+'/RIM-ONE/Cup/Glaucoma/*.png')
    files_fov=glob.glob(path_to_data+'/RIM-ONE/FOV/Glaucoma/*.png')
    path_to_crop_image=path_to_data+'/RIM-ONE/Images_crop/'
    path_to_crop_disc=path_to_data+'/RIM-ONE/Disc_crop/'
    path_to_crop_cup=path_to_data+'/RIM-ONE/Cup_crop/'
    
    num_of_img=len(files_img)
    pom=int(num_of_img*pocet_train)
    
    for i in range(len(files_img)):
        
        image=imread(files_img[i])
        fov=imread((files_fov[i])).astype(bool)
        mask_disc=imread(files_mask_disc[i])
        mask_cup=imread(files_mask_cup[i])
        
        if i<pom:
            center_new=Detection_of_disc(image,fov,sigma,size_of_erosion)
            output_crop_image, output_mask_disc,output_mask_cup=Crop_image(image,mask_disc,mask_cup,output_image_size,center_new)    
            output_mask_disc=output_mask_disc.astype(bool)
            output_mask_cup=output_mask_cup.astype(bool)   
            
            plt.imsave(path_to_crop_image + 'Train/' + files_img[i][60:],output_crop_image)
            plt.imsave(path_to_crop_disc + 'Train/' + files_mask_disc[i][58:],output_mask_disc)
            plt.imsave(path_to_crop_cup + 'Train/' + files_mask_cup[i][57:],output_mask_cup)
        else:
            plt.imsave(path_to_crop_image + 'Test/' + files_img[i][60:],image)
            plt.imsave(path_to_crop_disc + 'Test/' + files_mask_disc[i][58:],mask_disc)
            plt.imsave(path_to_crop_cup + 'Test/' + files_mask_cup[i][57:],mask_cup)
     
             
    # RIM-ONE - Healthy
    pocet_train=0.8
    
    files_img=glob.glob(path_to_data+'/RIM-ONE/Images/Healthy/*.png')
    files_mask_disc=glob.glob(path_to_data+'/RIM-ONE/Disc/Healthy/*.png')
    files_mask_cup=glob.glob(path_to_data+'/RIM-ONE/Cup/Healthy/*.png')
    files_fov=glob.glob(path_to_data+'/RIM-ONE/FOV/Healthy/*.png')
    path_to_crop_image=path_to_data+'/RIM-ONE/Images_crop/'
    path_to_crop_disc=path_to_data+'/RIM-ONE/Disc_crop/'
    path_to_crop_cup=path_to_data+'/RIM-ONE/Cup_crop/'
    
    num_of_img=len(files_img)
    pom=int(num_of_img*pocet_train)
    
    for i in range(len(files_img)):
        
        image=imread(files_img[i])
        fov=imread((files_fov[i])).astype(bool)
        mask_disc=imread(files_mask_disc[i])
        mask_cup=imread(files_mask_cup[i])
        
        if i<pom:
            center_new=Detection_of_disc(image,fov,sigma,size_of_erosion)
            output_crop_image, output_mask_disc,output_mask_cup=Crop_image(image,mask_disc,mask_cup,output_image_size,center_new)    
            output_mask_disc=output_mask_disc.astype(bool)
            output_mask_cup=output_mask_cup.astype(bool)   
            
            plt.imsave(path_to_crop_image + 'Train/' + files_img[i][59:],output_crop_image)
            plt.imsave(path_to_crop_disc + 'Train/' + files_mask_disc[i][57:],output_mask_disc)
            plt.imsave(path_to_crop_cup + 'Train/' + files_mask_cup[i][56:],output_mask_cup)
        else:
            plt.imsave(path_to_crop_image + 'Test/' + files_img[i][59:],image)
            plt.imsave(path_to_crop_disc + 'Test/' + files_mask_disc[i][57:],mask_disc)
            plt.imsave(path_to_crop_cup + 'Test/' + files_mask_cup[i][56:],mask_cup)
    
    # UOA-DR - Healthy
    pocet_train=0.8
    
    files_img=glob.glob(path_to_data+'/UoA_DR/Images/Healthy/*.png')
    files_mask_disc=glob.glob(path_to_data+'/UoA_DR/Disc/Healthy/*.png')
    files_mask_cup=glob.glob(path_to_data+'/UoA_DR/Cup/Healthy/*.png')
    files_fov=glob.glob(path_to_data+'/UoA_DR/FOV/Healthy/*.png')
    path_to_crop_image=path_to_data+'/UoA_DR/Images_crop/'
    path_to_crop_disc=path_to_data+'/UoA_DR/Disc_crop/'
    path_to_crop_cup=path_to_data+'/UoA_DR/Cup_crop/'
    
    num_of_img=len(files_img)
    pom=int(num_of_img*pocet_train)
    
    for i in range(len(files_img)):
        
        image=imread(files_img[i])
        fov=imread((files_fov[i])).astype(bool)
        mask_disc=imread(files_mask_disc[i])
        mask_cup=imread(files_mask_cup[i])
        
        if i<pom:
            center_new=Detection_of_disc(image,fov,sigma,size_of_erosion)
            output_crop_image, output_mask_disc,output_mask_cup=Crop_image(image,mask_disc,mask_cup,output_image_size,center_new)    
            output_mask_disc=output_mask_disc.astype(bool)
            output_mask_cup=output_mask_cup.astype(bool)   
            
            plt.imsave(path_to_crop_image + 'Train/' + files_img[i][58:],output_crop_image)
            plt.imsave(path_to_crop_disc + 'Train/' + files_mask_disc[i][56:],output_mask_disc)
            plt.imsave(path_to_crop_cup + 'Train/' + files_mask_cup[i][55:],output_mask_cup)
        else:
            plt.imsave(path_to_crop_image + 'Test/' + files_img[i][58:],image)
            plt.imsave(path_to_crop_disc + 'Test/' + files_mask_disc[i][56:],mask_disc)
            plt.imsave(path_to_crop_cup + 'Test/' + files_mask_cup[i][55:],mask_cup)
 
    
    # UOA-DR - NDPR
    pocet_train=0.8
    
    files_img=glob.glob(path_to_data+'/UoA_DR/Images/NDPR/*.png')
    files_mask_disc=glob.glob(path_to_data+'/UoA_DR/Disc/NDPR/*.png')
    files_mask_cup=glob.glob(path_to_data+'/UoA_DR/Cup/NDPR/*.png')
    files_fov=glob.glob(path_to_data+'/UoA_DR/FOV/NDPR/*.png')
    path_to_crop_image=path_to_data+'/UoA_DR/Images_crop/'
    path_to_crop_disc=path_to_data+'/UoA_DR/Disc_crop/'
    path_to_crop_cup=path_to_data+'/UoA_DR/Cup_crop/'
    
    num_of_img=len(files_img)
    pom=int(num_of_img*pocet_train)
    
    for i in range(len(files_img)):
        
        image=imread(files_img[i])
        fov=imread((files_fov[i])).astype(bool)
        mask_disc=imread(files_mask_disc[i])
        mask_cup=imread(files_mask_cup[i])
        
        if i<pom:
            center_new=Detection_of_disc(image,fov,sigma,size_of_erosion)
            output_crop_image, output_mask_disc,output_mask_cup=Crop_image(image,mask_disc,mask_cup,output_image_size,center_new)    
            output_mask_disc=output_mask_disc.astype(bool)
            output_mask_cup=output_mask_cup.astype(bool)   
            
            plt.imsave(path_to_crop_image + 'Train/' + files_img[i][55:],output_crop_image)
            plt.imsave(path_to_crop_disc + 'Train/' + files_mask_disc[i][53:],output_mask_disc)
            plt.imsave(path_to_crop_cup + 'Train/' + files_mask_cup[i][52:],output_mask_cup)
        else:
            plt.imsave(path_to_crop_image + 'Test/' + files_img[i][55:],image)
            plt.imsave(path_to_crop_disc + 'Test/' + files_mask_disc[i][53:],mask_disc)
            plt.imsave(path_to_crop_cup + 'Test/' + files_mask_cup[i][52:],mask_cup)
    
    # UOA-DR - PDR
    pocet_train=0.8
    
    files_img=glob.glob(path_to_data+'/UoA_DR/Images/PDR/*.png')
    files_mask_disc=glob.glob(path_to_data+'/UoA_DR/Disc/PDR/*.png')
    files_mask_cup=glob.glob(path_to_data+'/UoA_DR/Cup/PDR/*.png')
    files_fov=glob.glob(path_to_data+'/UoA_DR/FOV/PDR/*.png')
    path_to_crop_image=path_to_data+'/UoA_DR/Images_crop/'
    path_to_crop_disc=path_to_data+'/UoA_DR/Disc_crop/'
    path_to_crop_cup=path_to_data+'/UoA_DR/Cup_crop/'
    
    num_of_img=len(files_img)
    pom=int(num_of_img*pocet_train)
    
    for i in range(len(files_img)):
        
        image=imread(files_img[i])
        fov=imread((files_fov[i])).astype(bool)
        mask_disc=imread(files_mask_disc[i])
        mask_cup=imread(files_mask_cup[i])
        
        if i<pom:
            center_new=Detection_of_disc(image,fov,sigma,size_of_erosion)
            output_crop_image, output_mask_disc,output_mask_cup=Crop_image(image,mask_disc,mask_cup,output_image_size,center_new)    
            output_mask_disc=output_mask_disc.astype(bool)
            output_mask_cup=output_mask_cup.astype(bool)   
            
            plt.imsave(path_to_crop_image + 'Train/' + files_img[i][54:],output_crop_image)
            plt.imsave(path_to_crop_disc + 'Train/' + files_mask_disc[i][52:],output_mask_disc)
            plt.imsave(path_to_crop_cup + 'Train/' + files_mask_cup[i][51:],output_mask_cup)
        else:
            plt.imsave(path_to_crop_image + 'Test/' + files_img[i][54:],image)
            plt.imsave(path_to_crop_disc + 'Test/' + files_mask_disc[i][52:],mask_disc)
            plt.imsave(path_to_crop_cup + 'Test/' + files_mask_cup[i][51:],mask_cup)
    
 
    