import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet18 
from sparsity_module_v1 import Sparse_learn
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# CIFAR10 dataset 
class Dataset_cifar100:
    def __init__(self):
        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])

        

        self.train_set = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=100, shuffle=True, num_workers=2)

        self.test_set = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=self.transform)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=100, shuffle=False, num_workers=2)



if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    dataset=Dataset_cifar100()
    model=resnet18(num_classes=100).to(device)
    model.train()
    epochs=500
    learning_rate=0.1
    sparsity=0.5
    sparse=Sparse_learn()
    result,measure_value=sparse.fit_sparse(model,dataset,epochs,learning_rate,device,sparsity)
    data_collected=pd.DataFrame(result,columns=['epoch','training_loss','test_loss','test_accuracy'])
    plt.plot(data_collected['epoch'], data_collected['test_accuracy'], label='Our approach')
    plt.xlabel(xlabel='Epochs')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy graph')
    plt.legend()
    plt.show()



