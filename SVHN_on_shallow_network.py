#code for training a shallow ann network on SVHN with sparsification


import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.nn.utils import prune                                        # To create binary masks for each layer that will act as prehook during training
import pandas as pd
import pickle
from sparsity_module import Sparse_learn
get_ipython().run_line_magic('matplotlib', 'inline')




class Svhn_Dataset:
    def __init__(self, batch_size):
        self.transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.0406], std=[0.229, 0.224, 0.225])])
        self.train_set = torchvision.datasets.SVHN(
            root='./data1', split='train', download=True, transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=batch_size, shuffle=True, num_workers=2)

        self.test_set = torchvision.datasets.SVHN(
            root='./data1', split='test', download=True, transform=self.transform)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=batch_size, shuffle=False, num_workers=2)





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




if __name__=="__main__":
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # architecture specs for shallow network
    input_size = 3072                              # 32x32x3----for SVHN dataset
    hidden_size = 1000                            
    num_classes = 10
    epochs=100
    learning_rate=0.1
    sparsity=0.9
    batch_size=100
    dataset=Svhn_Dataset(batch_size)
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







