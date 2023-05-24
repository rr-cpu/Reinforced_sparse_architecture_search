import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.nn.utils import prune 
import pandas as pd
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')



class Sparse_learn:
    def __init__(self):
        pass
    
    def fit_sparse(self, model, dataset, epochs, lr ,device,sparsity_value, opt_func=torch.optim.SGD):
        history = []
        loss_previous=0
        optimizer = opt_func(model.parameters(), lr)

    #     Initial random pruning
        for name, module in model.named_modules():
            if isinstance(module,torch.nn.Conv2d) or isinstance(module,torch.nn.Linear):
                prune.random_unstructured(module, name='weight', amount=sparsity_value)


    #     Initialize the measure values to 0
        measure_value={}
        for i,_ in model.named_parameters():
            if("weight" in i) and ("mask" not in i):
                measure_value[i]=torch.zeros(model.state_dict()[i].size()).to(device)

    #     Training Phase
        for epoch in range(epochs):     
            for i,(images,labels) in enumerate(dataset.train_loader):
                images,labels=images.to(device),labels.to(device)

                output=model(images)
                loss = F.cross_entropy(output, labels)
                optimizer.zero_grad()
                loss.backward()

    #           calculating the change in loss(i.e., Reward)
                loss_change=loss.item()-loss_previous
                loss_previous=loss.item()
    #           value updation for each layer
                for l,_ in model.named_parameters():
                    if('weight' in l) and ('mask' not in l):
                        mask=dict(model.named_buffers())[l.replace('orig','mask')]
                        measure_value[l].add_(mask, alpha=0.01*loss_change)
    #           update the weights
                optimizer.step()
    #           update the sparse architecture
                if i%10==0 and epoch<epochs-1:
                    for l,_ in model.named_parameters():
                        if('weight' in l) and ('mask' not in l):
                            mask=dict(model.named_buffers())[l.replace('orig','mask')]
                            mask[:,:]=1
                            neurons_count=math.floor((torch.numel(mask)*(1-sparsity_value)))
                            exploitation_count=math.floor(neurons_count*(epoch/epochs))
                            exploration_count=neurons_count-exploitation_count
                            if exploitation_count!=0:
                                topk = torch.topk(measure_value[l].view(-1), k=(exploitation_count), largest=True)
                                mask.view(-1)[topk.indices] = 0
                                mask[::]=torch.nn.utils.prune.RandomUnstructured(amount=(torch.numel(mask)-neurons_count)).prune(mask)
                                mask.view(-1)[topk.indices] = 1
                            else:
                                mask[::]=torch.nn.utils.prune.RandomUnstructured(amount=(torch.numel(mask)-neurons_count)).prune(mask)
                elif(epoch==epochs-1 and i==0):
                    for l,_ in model.named_parameters():
                        if('weight' in l) and ('mask' not in l):
                            mask=dict(model.named_buffers())[l.replace('orig','mask')]
                            mask[:,:]=0
                            neurons_count=math.floor((torch.numel(mask)*(1-sparsity_value)))
                            topk = torch.topk(measure_value[l].view(-1), k=(neurons_count), largest=True)
                            mask.view(-1)[topk.indices] = 1


    #           Validation phase
                
            val_loss,val_acc = self.evaluate(model, dataset.test_loader, device)
            print(f"epoch: {epoch}, train_loss: {loss.item():.4f}, test_loss: {val_loss.item():.4f}, test_acc:{val_acc:.4f} ")
            history.append([epoch,loss.item(),val_loss.item(),val_acc])
        return history,measure_value
    
    def evaluate(self,model, test_loader,device):
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
           # print(images.view(128,-1).shape)
            outputs = model(images)
            eval_loss = F.cross_entropy(outputs, labels)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        return eval_loss, acc

