Deep learning teaching kit.
===

## Start-ups

#### Torch installation
    
Please refer to the official torch webpage: [here](http://torch.ch/docs/getting-started.html) 

For Windows users, please refer to [Running Torch on Windows](https://github.com/torch/torch7/wiki/Windows#using-a-virtual-machine).

#### CUDA support

Thanks to the rapid develop of modern GPUs, training deep networks in a large-scale data-driven context becomes more and more affordable in terms of both time and resource cost.
Training networks on an GPU is recommended for the assignments in all the lab assignments in this kit, which applies to both convolutional network (in lab1 and lab2) as well as the recurrent networks (in lab4).

Practically, we urge students to install CUDA libraries prior to install any deep learning specific tools.
To install them, please refer to:

- [CUDA installation page](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4RK3pacJh)
- [CUDNN torch bindings](https://github.com/soumith/cudnn.torch)


#### GIT basic usage

For instance, in order to obtain the kit on a laptop or workstation, and further access assignment 1:
```
    git clone git@bitbucket.org:junbo_jake_zhao/deeplearningkit.git
    cd a1
```