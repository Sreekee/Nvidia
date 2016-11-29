Deep Learning Teaching Kit Labs
===

## System Requirements

#### NVIDIA CUDA/GPUs

Thanks to the rapid develop of modern GPUs, training deep networks in a large-scale, data-driven context becomes more and more affordable in terms of both time and resource cost.
Training neural networks on GPUs is highly recommended for the Deep Learning Teaching Kit labs, including both convolutional networks (Lab1 and Lab2) and recurrent networks (in Lab4).

**You must have an [NVIDIA CUDA Capable GPU](https://developer.nvidia.com/cuda-gpus)
to use the compiled binaries.**

The labs in the teaching kit require a CUDA supported operating system,
C compiler, and a recent CUDA Toolkit. The CUDA Toolkit can be downloaded
from the [CUDA Download](https://developer.nvidia.com/cuda-downloads) page.
Instructions on how to install the CUDA Toolkit are available in the
[Quick Start page](http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).
Installation guides and the list of supported C compilers for [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html),
[Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), and
[OSX](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html) are
also found in the [CUDA Toolkit Documentation Page](http://docs.nvidia.com/cuda/index.html).

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