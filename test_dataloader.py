# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 18:42:42 2022

@author: nohel
"""

from funkce_uprava_dataloaderu import DataLoader
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":  
    batch=1
    
    loader=DataLoader(sigma=60,size_of_erosion=80,output_image_size=[768,768])
    trainloader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=True)
    
    for it,(data,mask) in enumerate(trainloader): ### you can iterate over dataset (one epoch)
        if it==0:
            plt.figure(figsize=[10,10])
            plt.subplot(1,2,1)
            plt.imshow(data[0,0,:,:].detach().cpu().numpy(),cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(mask[0,0,:,:].detach().cpu().numpy())
            plt.show()
            break
    