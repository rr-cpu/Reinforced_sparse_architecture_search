#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from sparsity_module import Sparse_learn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


class Svhn_Dataset:
    def __init__(self):
        self.transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.0406], std=[0.229, 0.224, 0.225])])
        self.train_set = torchvision.datasets.SVHN(
            root='./data1', split='train', download=True, transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=128, shuffle=True, num_workers=2)

        self.test_set = torchvision.datasets.SVHN(
            root='./data1', split='test', download=True, transform=self.transform)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=128, shuffle=False, num_workers=2)

    def imshow(self,img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    def show_images(self):        
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)
        print(images.shape,labels.shape)
        # show images
        self.imshow(torchvision.utils.make_grid(images))
        # print labels


# In[ ]:


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()        
        self.l2 = nn.Linear(hidden_size,num_classes)
       
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


# In[ ]:


if __name__=="__main__":
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset=Svhn_Dataset()
    dataset.show_images()
    # architecture specs for shallow network
    input_size = 3072                              # 32x32x3----for SVHN dataset
    hidden_size = 1000                            
    num_classes = 10
    epochs=100
    learning_rate=0.1
    sparsity=0.9
    model=NeuralNet(input_size, hidden_size, num_classes).to(device)
    sparse=Sparse_learn()
    result,measure_value=sparse.fit_sparse(model,dataset,epochs,learning_rate,device,sparsity)
    data_collected=pd.DataFrame(result,columns=['epoch','training_loss','test_loss','test_accuracy'])
    plt.plot(data_collected['epoch'], data_collected['test_accuracy'], label='Our approach')
    plt.xlabel(xlabel='Epochs')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy graph')
    plt.legend()
    plt.show()


# In[ ]:




