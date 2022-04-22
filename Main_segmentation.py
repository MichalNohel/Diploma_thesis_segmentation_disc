# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:18:29 2022

@author: nohel
"""

import numpy as np
from funkce_final import DataLoader, Unet, dice_loss
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from IPython.display import clear_output

if __name__ == "__main__": 
    #Parameters
    lr=0.001
    epochs=25
    batch=16
    threshold=0.5
    color_preprocesing="HSV"
    segmentation_type="disc"
    
    
    loader=DataLoader(split="Train",path_to_data="D:\Diploma_thesis_segmentation_disc/Data_500_500",color_preprocesing=color_preprocesing,segmentation_type=segmentation_type)
    trainloader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=True)