# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:18:29 2022

@author: nohel
"""

import numpy as np
from funkce_final import DataLoader, Unet, dice_loss, dice_coefficient
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from IPython.display import clear_output
from skimage.color import hsv2rgb

if __name__ == "__main__": 
    #Parameters
    lr=0.001
    epochs=25
    batch=8
    threshold=0.5
    color_preprocesing="HSV"
    segmentation_type="disc"
    
    
    loader=DataLoader(split="Train",path_to_data="D:\Diploma_thesis_segmentation_disc/Data_500_500",color_preprocesing=color_preprocesing,segmentation_type=segmentation_type)
    trainloader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=True)
    
    net=Unet().cuda()    
    optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=1e-8)
    sheduler=StepLR(optimizer,step_size=7, gamma=0.1) #decreasing of learning rate
    
    train_loss = []
    #test_loss = []
    train_acc = []  
    #test_acc = []
    train_dice = []
    
    it=-1
    for epoch in range(epochs):
        acc_tmp = []
        loss_tmp = []
        dice_tmp = []
        print('epoch number ' + str(epoch+1))
        
        for k,(data,lbl) in enumerate(trainloader):
            it+=1
            data=data.cuda()
            lbl=lbl.cuda() 
            
            net.train()
            output=net(data)
            
            output=torch.sigmoid(output)
            
            #loss = -torch.mean(lbl*torch.log(output)+(1-lbl)*torch.log(1-output))
            #loss = -torch.mean(20*lbl*torch.log(output)+1*(1-lbl)*torch.log(1-output)) #vahovaní
            loss=dice_loss(lbl,output)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            lbl_mask=lbl.detach().cpu().numpy()
            output_mask=output.detach().cpu().numpy() > threshold
            
            acc=np.mean((output_mask==lbl_mask))
            acc_tmp.append(acc)
            loss_tmp.append(loss.cpu().detach().numpy())
            
            dice_tmp.append(dice_coefficient(output_mask,lbl_mask))
            
            if (it % 30==0):
                clear_output()
                plt.figure(figsize=[10,10])
                plt.plot(acc_tmp,label='train acc')
                plt.plot(loss_tmp,label='train loss')
                plt.plot(dice_tmp,label='dice')
                plt.legend(loc="upper left")
                plt.title('train')
                plt.show()
                print('iteration ' + str(it))
            
        train_loss.append(np.mean(loss_tmp))
        train_acc.append(np.mean(acc_tmp))
        train_dice.append(np.mean(dice_tmp)) 
        
        
        sheduler.step()
        
        clear_output()
        plt.figure(figsize=[10,10])
        plt.plot(train_acc,label='train acc')
        plt.plot(train_loss,label='train loss')
        plt.plot(train_dice,label='dice')
        plt.legend(loc="upper left")
        plt.title('train')
        plt.show()
    
        plt.figure(figsize=[10,10])
        plt.subplot(1,3,1)        
        im_pom=data[0,:,:,:].detach().cpu().numpy()   
        im_pom=np.transpose(im_pom,(1,2,0))
        im_pom=hsv2rgb(im_pom)
        plt.imshow(im_pom)
        
        
        plt.subplot(1,3,2)    
        plt.imshow(lbl[0,0,:,:].detach().cpu().numpy())
        
        plt.subplot(1,3,3)    
        plt.imshow(output[0,0,:,:].detach().cpu().numpy()>threshold)
        plt.show() 
            
            
            
            
        
        
        
    
    
    