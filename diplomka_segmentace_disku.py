# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 15:23:20 2021

@author: nohel
"""
import numpy as np
from funkce import DataLoader, Unet, dice_loss, rekonstrukce_masky_z_patchu
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from IPython.display import clear_output

if __name__ == "__main__":  
    
    lr=0.001
    epochs=25
    batch=16
    threshold=0.5
    predzpracovani="HSV"
    
    loader=DataLoader(split="Train",predzpracovani=predzpracovani)
    trainloader=torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=True)
    '''
    loader=DataLoader(split="Validation")
    validationloader=torch.utils.data.DataLoader(loader,batch_size=1, num_workers=0, shuffle=False)
    '''
    
    loader=DataLoader(split="Test",predzpracovani=predzpracovani)
    testloader=torch.utils.data.DataLoader(loader,batch_size=1, num_workers=0, shuffle=False)
    
    net=Unet().cuda()
    
    optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=1e-8)
    sheduler=StepLR(optimizer,step_size=7, gamma=0.1) #snižování learning rate
    
    train_loss = []
    test_loss = []
    train_acc = []  
    test_acc = []
    
    it=-1
    for epoch in range(epochs):
        acc_tmp = []
        loss_tmp = []
        for k,(data,lbl) in enumerate(trainloader):
            it+=1
            
            data = data.cuda()
            lbl = lbl.cuda()
            
            net.train()
            output = net(data)
            
            output=torch.sigmoid(output)
            
            #loss = -torch.mean(lbl*torch.log(output)+(1-lbl)*torch.log(1-output))
            #loss = -torch.mean(20*lbl*torch.log(output)+1*(1-lbl)*torch.log(1-output)) #vahovaní
            loss=dice_loss(lbl,output)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            lbl_num=lbl.detach().cpu().numpy()
            clas=output.detach().cpu().numpy() > threshold
            
            acc=np.mean((clas==lbl_num))
            acc_tmp.append(acc)
            loss_tmp.append(loss.cpu().detach().numpy())
            
            if (it % 100==0):
                clear_output()
                plt.figure(figsize=[10,10])
                plt.plot(acc_tmp,label='train acc')
                plt.legend(loc="upper left")
                plt.title('train_acc')
                plt.show()
                
                plt.figure(figsize=[10,10])
                plt.plot(loss_tmp,label='train loss')
                plt.legend(loc="upper left")
                plt.title('train_loss')
                plt.show()                
                print(it)
                
            
            
        
        train_loss.append(np.mean(loss_tmp))
        train_acc.append(np.mean(acc_tmp))
        
        acc_tmp = []
        loss_tmp = []
        
        for kk,(data,lbl) in enumerate(testloader):
            with torch.no_grad():
                data = data.cuda()              
                lbl = lbl.cuda()              
                net.eval()
    
                output = net(data)
                output=torch.sigmoid(output)
    
                #loss = -torch.mean(lbl*torch.log(output)+(1-lbl)*torch.log(1-output))
                #loss = -torch.mean(20*lbl*torch.log(output)+1*(1-lbl)*torch.log(1-output)) #vahovaní
                loss=dice_loss(lbl,output)
              
    
                lbl_num=lbl.detach().cpu().numpy()
                clas=output.detach().cpu().numpy() > threshold
    
                acc=np.mean((clas==lbl_num))
                acc_tmp.append(acc)
                loss_tmp.append(loss.cpu().detach().numpy())
        
        test_loss.append(np.mean(loss_tmp))
        test_acc.append(np.mean(acc_tmp))
        
        sheduler.step()
        
        clear_output()
        plt.figure(figsize=[10,10])
        plt.plot(train_acc,label='train acc')
        plt.plot(test_acc,label='test acc')
        plt.legend(loc="upper left")
        plt.title('acc')
        plt.show()
    
        plt.figure(figsize=[10,10])
        plt.plot(train_loss,label='train loss')
        plt.plot(test_loss,label='test loss')
        plt.legend(loc="upper left")
        plt.title('loss')
        plt.show()
    
        plt.figure(figsize=[10,10])
        plt.subplot(1,3,1)
        plt.imshow(data[0,0,:,:].detach().cpu().numpy())
        
        plt.subplot(1,3,2)    
        plt.imshow(lbl[0,0,:,:].detach().cpu().numpy())
        
        plt.subplot(1,3,3)    
        plt.imshow(output[0,0,:,:].detach().cpu().numpy()>threshold)
        plt.show() 
    
    
    # %% vykresleni a ulozeni vysledku
    ZAPISOVAT=1
    if ZAPISOVAT:
        pocitadlo=0
        pocitadlo_obrazku=1
        dataset_output=[]
        dataset_lbl=[]
        for kk,(data,lbl) in enumerate(testloader):
            with torch.no_grad():
                data=data.cuda()
                lbl=lbl.cuda()
                net.eval()
                output = net(data)
                output=torch.sigmoid(output)
                    
                lbl_num=lbl.detach().cpu().numpy()
                clas=output.detach().cpu().numpy() > threshold
                    
                dataset_output.append(clas[0,0,:,:])
                dataset_lbl.append(lbl_num[0,0,:,:])
                pocitadlo=pocitadlo+1
                if pocitadlo==169:
                    rekonstrukce_output,rekonstrukce_lbl=rekonstrukce_masky_z_patchu(dataset_output,dataset_lbl)
                    plt.figure()
                    plt.subplot(1,2,1)
                    plt.imshow(rekonstrukce_lbl)
                    plt.title('Originální maska disku')
                    plt.subplot(1,2,2)
                    plt.imshow(rekonstrukce_output)
                    plt.title('Výstup ze sítě')
                    #plt.savefig("C:/Users/nohel/Desktop/Diplomka_data/Sedoton_25_epoch_uceni/test_" + str(pocitadlo_obrazku))
                    plt.show()
                    
                    pocitadlo_obrazku=pocitadlo_obrazku+1
                    pocitadlo=0
                    dataset_output=[]
                    dataset_lbl=[]
                    
            
        
    
    
    
    
    
    
    
    
    
    
    
    