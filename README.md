Deep Learning Teaching Kit Labs
===

## System Requirements

#### NVIDIA CUDA/GPUs

Thanks to the rapid develop of modern GPUs, training deep networks in a large-scale, data-driven context becomes more and more affordable in terms of both time and resource cost.
Training neural networks on GPUs is recommended for the Deep Learning Teaching Kit labs, including both convolutional networks (Lab1 and Lab2) and recurrent networks (in Lab4).

Practically, we urge students to install CUDA libraries prior to install any deep learning specific tools.
To install them, please refer to:

- [CUDA installation page](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4RK3pacJh)
- [CUDNN torch bindings](https://github.com/soumith/cudnn.torch)

#### Torch installation
    
Please refer to the official torch webpage: [here](http://torch.ch/docs/getting-started.html) 

For Windows users, please refer to [Running Torch on Windows](https://github.com/torch/torch7/wiki/Windows#using-a-virtual-machine).
#### GIT basic usage

For instance, in order to obtain the kit on a laptop or workstation, and further access assignment 1:
```
    git clone git@bitbucket.org:junbo_jake_zhao/deeplearningkit.git
    cd a1
```