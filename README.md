# Reinforced Sparse Architecture Search
This repository contains the code for the paper titled "Reinforced Sparse Architecture Search" submitted to NeurIPS 2023 conference.

Reinforced Sparse Architecture is an unstructured iterative pruning approach where pruning is performed based on the measure_value that is used as a importance metric for the parameters. The measure_value is learned during training. please refer to the paper for more details.

we have performed experimentation using shallow network on MNIST and SVHN and using Resnet18 and VGG16 on Cifar10 and Cifar100 datasets. We have used the pytorch framework for our implementation. We have used the prune module of Pytorch for creating masks which will act as prehook during training.

![Screenshot from 2023-05-13 16-01-55](https://github.com/rr-cpu/Reinforced_sparse_architecture_search/assets/56760930/c56d3073-0221-4375-b3a7-e65a245b8385)
