# Reinforced Sparse Architecture Search
This repository contains the code for the paper titled "Reinforced Sparse Architecture Search" submitted to NeurIPS 2023 conference.

![Screenshot from 2023-05-13 16-01-55](https://github.com/rr-cpu/Reinforced_sparse_architecture_search/assets/56760930/c56d3073-0221-4375-b3a7-e65a245b8385)

Reinforced Sparse Architecture is an unstructured iterative pruning approach where pruning is performed based on the measure_value that is used as a importance metric for the parameters. The measure_value is learned during training. Please refer to the paper for more details.

We have performed experimentation using shallow network on [MNIST](MNIST on shallow network.py) and [SVHN](SVHN on shallow network.py) and using Resnet18 and VGG16 on Cifar10 and Cifar100 datasets. We have used the pytorch framework for our implementation. We have used the prune module of Pytorch for creating masks which will act as pre-hook during training. For updating the architecture, we update the pre-hook mask.

For reproducing the results, download the files and run the .py files corresponding to the model and dataset name you wish to train. The hyperparameters are all preset to the values mentioned in the paper. Make sure the [sparsity_module.py](sparsity_module.py) file is in same folder as your main running file since it contains the fit_sparse() fuction to train the sparse model.
          
