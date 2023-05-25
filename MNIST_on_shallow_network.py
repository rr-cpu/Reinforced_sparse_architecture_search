
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
from sparsity_module import Sparse_learn
get_ipython().run_line_magic('matplotlib', 'inline')




# Preparing the data
# creating a dataset class with train_loader and test_loader, with batch_size=100 by default
class Dataset_MNIST:
    def __init__(self,batch_size=100):
        self.train_dataset = torchvision.datasets.MNIST(root='./data', 
                                                   train=True, 
                                                   transform=transforms.ToTensor(),  
                                                   download=True)

        self.test_dataset = torchvision.datasets.MNIST(root='./data', 
                                                  train=False, 
                                                  transform=transforms.ToTensor())

        # Data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, 
                                                  batch_size=batch_size, 
                                                  shuffle=False)



        




# Creating the shallow architecture with one hidden layer
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
    dataset=Dataset_MNIST(batch_size=100)
    # architecture specs for shallow network
    input_size = 784                              # 28x28----for MNIST dataset
    hidden_size = 1000                            
    num_classes = 10
    epochs=100
    learning_rate=0.1
    sparsity=0.9
    model=NeuralNet(input_size, hidden_size, num_classes).to(device)        
    sparse=Sparse_learn()
    result,measure_value=sparse.fit_sparse(model,dataset,epochs,learning_rate,device,sparsity)
    data_collected=pd.DataFrame(result,columns=['epoch','training_loss','test_loss','test_accuracy'])
    plt.plot(data_collected['epoch'], data_collected['test_accuracy'], color='r', label='Our approach')
    plt.xlabel(xlabel='Epochs')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy graph')
    plt.legend()
    plt.show()





