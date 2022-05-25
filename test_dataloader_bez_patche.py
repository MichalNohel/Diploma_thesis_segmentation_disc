# -*- coding: utf-8 -*-
"""
Created on Wed May 25 08:54:19 2022

@author: nohel
"""


from funkce_final_bez_patche import DataLoader,Unet,Postprocesing
import torch
import matplotlib.pyplot as plt
import numpy as np 


    

if __name__ == "__main__":  
    batch=1 
    threshold=0.5
    color_preprocesing="RGB"
    segmentation_type="disc"
    output_size=(int(608),int(608),int(3))
    #path_to_data="D:\Diploma_thesis_segmentation_disc/Data_500_500"
    path_to_data="D:\Diploma_thesis_segmentation_disc/Data_640_640"
    
    loader=DataLoader(split="Train",path_to_data=path_to_data,color_preprocesing=color_preprocesing,segmentation_type=segmentation_type,output_size=output_size)
    trainloader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=True)
    
    batch=1
    loader=DataLoader(split="Test",path_to_data=path_to_data,color_preprocesing=color_preprocesing,segmentation_type=segmentation_type,output_size=output_size)
    testloader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=False)
    
    loader=DataLoader(split="HRF",path_to_data="D:\Diploma_thesis_segmentation_disc",color_preprocesing=color_preprocesing,segmentation_type=segmentation_type,output_size=output_size)
    HRF_loader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=False)
    
    net = Unet().cuda()   
    net.load_state_dict(torch.load("model_01_RGB_bez_patche.pth"))
    net.eval()
    
    pom="HRF"
    
    if pom=="Test":
        for kk,(data,mask,img_orig,disc_orig,cup_orig,coordinates) in enumerate(testloader):
            with torch.no_grad():
                if (kk % 10==0): 
                    net.eval()  
                    data=data.cuda()
                    output=net(data)
                    output=torch.sigmoid(output)
                    output=output.detach().cpu().numpy() > threshold
                    
                    pom_sourad=coordinates.detach().cpu().numpy()[0]               
                    output_mask=np.zeros([disc_orig.shape[1],disc_orig.shape[2]])       
                    output_mask_final=np.zeros([disc_orig.shape[1],disc_orig.shape[2]])   
                    
                    if (pom_sourad[1]-int(output_size[0]/2)<0):
                        x_start=0
                    elif((pom_sourad[1]+int(output_size[0]/2))>output_mask.shape[0]):
                        x_start=output_mask.shape[0]-output_size[0]
                    else:
                        x_start=pom_sourad[1]-int(output_size[0]/2)
                        
                    if (pom_sourad[0]-int(output_size[0]/2)<0):
                        y_start=0
                    elif((pom_sourad[0]+int(output_size[0]/2))>output_mask.shape[1]):
                        y_start=output_mask.shape[1]-output_size[0]
                    else:
                        y_start=pom_sourad[0]-int(output_size[0]/2)
                        
                        
        
                    output_mask[x_start:x_start+output_size[0],y_start:y_start+output_size[0]]=output[0,0,:,:]
                    output_mask=output_mask.astype(bool)
                    
                    # Postprocesing
                    '''
                    if (kk==25):
                        pom=1
                    '''
                    min_size_of_disk=2000
                    size_of_disk_for_erosion=30
                    ploting=1
                    
                    output_final=Postprocesing(output[0,0,:,:],min_size_of_disk,size_of_disk_for_erosion,ploting)
                    
                    output_mask_final[x_start:x_start+output_size[0],y_start:y_start+output_size[0]]=output_final
                    output_mask_final=output_mask_final.astype(bool)
                    
                    disc_orig=disc_orig[0,:,:].detach().cpu().numpy() 
                
                    plt.figure(figsize=[10,10])
                    plt.subplot(2,4,1)        
                    im_pom=img_orig[0,:,:,:].detach().cpu().numpy()/255   
                    plt.imshow(im_pom) 
                    plt.title(str(kk))
                    
                    plt.subplot(2,4,2)    
                    plt.imshow(disc_orig)
                    plt.title('Orig maska')
                    
                    plt.subplot(2,4,3)    
                    plt.imshow(output_mask)
                    plt.title('Vystup sítě')
                    
                    plt.subplot(2,4,4)    
                    plt.imshow(output_mask_final)
                    plt.title('Postprocesing')
                    
                    plt.subplot(2,4,5)        
                    data_pom=data[0,:,:,:].detach().cpu().numpy()/255  
                    data_pom=np.transpose(data_pom,(1,2,0))
                    #data_pom=hsv2rgb(data_pom)
                    plt.imshow(data_pom)  
                    
                    plt.subplot(2,4,6)                       
                    plt.imshow(mask[0,0,:,:].detach().cpu().numpy())
                    
                    plt.subplot(2,4,7)                       
                    plt.imshow(output[0,0,:,:])
                    
                    plt.subplot(2,4,8)                       
                    plt.imshow(output_final)
                    
                    plt.show() 
                    
                    print('Test - iteration ' + str(kk))
 
                
    if pom=="HRF":         
        for it,(data,img_orig,coordinates) in enumerate(HRF_loader): ### you can iterate over dataset (one epoch)
            if it<50:
                net.eval() 
                data=data.cuda()
                output=net(data)
                output=torch.sigmoid(output)
                output=output.detach().cpu().numpy() > threshold
                    
                pom_sourad=coordinates.detach().cpu().numpy()[0]               
                output_mask=np.zeros([img_orig.shape[1],img_orig.shape[2]])                
                    
                if (pom_sourad[1]-int(output_size[0]/2)<0):
                    x_start=0
                elif((pom_sourad[1]+int(output_size[0]/2))>output_mask.shape[0]):
                    x_start=output_mask.shape[0]-output_size[0]
                else:
                    x_start=pom_sourad[1]-int(output_size[0]/2)
                        
                if (pom_sourad[0]-int(output_size[0]/2)<0):
                    y_start=0
                elif((pom_sourad[0]+int(output_size[0]/2))>output_mask.shape[1]):
                    y_start=output_mask.shape[1]-output_size[0]
                else:
                    y_start=pom_sourad[0]-int(output_size[0]/2)
                        
                        
    
                output_mask[x_start:x_start+output_size[0],y_start:y_start+output_size[0]]=output[0,0,:,:]
                output_mask=output_mask.astype(bool)
                    
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
                plt.imshow(output[0,0,:,:])            
                plt.show()    
       
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    