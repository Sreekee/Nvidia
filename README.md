# Deep Learning Teaching Kit Lab/solution Repository

Welcome to the Deep Learning Teaching Kit Lab/solution repository. The kit and associated labs are produced jointly by NVIDIA and New York University (NYU).  All material is available under the [Creative Commons Attribution-NonCommercial License](http://creativecommons.org/licenses/by-nc/4.0/).

The labs are designed to be open-ended, multidisciplinary, one- to three-week programming and written assignments for students. Each lab contains a description, pointers to required sample code, and suggestions on how to evaluate and have students submit solutions. Some programming labs include optimizing a neural network training model and suggest students submit solutions to Kaggle using [Kaggle In Class](https://inclass.kaggle.com/) to compare performance against benchmarks. Please read the Kaggle In Class [FAQ](https://www.kaggle.com/wiki/KaggleInClass) for more information on how to set up your course using Kaggle. Students submit to Kaggle is **not** a requirement to make use of the labs.

This repository contains lab solutions developed by actual students who have taken the Deep Learning curriculum course at NYU. Some solutions may require additional, undocumented instructions to properly execute, and may not run perfectly “out-of-box”. Still, they should be useful examples of successful solutions using a variety of techniques.

## System Requirements

#### NVIDIA CUDA/GPUs

Thanks to the rapid development of modern GPUs, training deep networks in a large-scale, data-driven context becomes more and more affordable in terms of both time and resource cost.
Training neural networks on [NVIDIA CUDA-enabled GPUs](https://developer.nvidia.com/cuda-gpus) is practically a necessity for the Teaching Kit labs, including both convolutional networks (Lab1 and Lab2) and recurrent networks (in Lab4).

The use of GPUs for the Teaching Kit labs requires a CUDA supported operating system, C compiler, and a recent CUDA Toolkit. The CUDA Toolkit can be downloaded
from the [CUDA Download](https://developer.nvidia.com/cuda-downloads) page. Instructions on how to install the CUDA Toolkit are available in the
[Quick Start page](http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html). Installation guides and the list of supported C compilers for [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html), [Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), and
[OSX](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html) are also found in the [CUDA Toolkit Documentation Page](http://docs.nvidia.com/cuda/index.html).

CUDA and the associated libraries should be installed prior to any deep learning specific tools.

#### Torch computing framework
    
The Deep Learning Teaching Kit labs and example solutions are based on the [Torch](http://torch.ch) computing framework. Please refer to [Getting started with Torch](http://torch.ch/docs/getting-started.html) for instruction on Torch installation, examples and documentation.

For Windows users, please refer to [Running Torch on Windows](https://github.com/torch/torch7/wiki/Windows#using-a-virtual-machine).

#### cuDNN

The NVIDIA CUDA Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.

To install cuDNN and use cuDNN with Torch, please follow the README on the [cuDNN torch bindings](https://github.com/soumith/cudnn.torch) project.

## About the labs and solutions

#### Recommended release time of the labs
* Lab1: finishing at least slides module 1-prepare and 2-dl-basics
* Lab2: finishing slides module 3-convnets
* Lab3: finishing slides module 4-energy-based-learning and 5-optimization
* Lab4: finishing slides module 6-learning-with-memory.


## Compiling and running lab solutions

#### GIT basics

As an example, to clone the labs/solutions on your machine and access Lab1:
```
    git clone git@bitbucket.org:junbo_jake_zhao/deeplearningkit.git
    cd Lab1
```