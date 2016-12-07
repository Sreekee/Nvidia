# Deep Learning Teaching Kit Lab/solution Repository

Welcome to the Deep Learning Teaching Kit Lab/solution repository. The kit and associated labs are produced jointly by NVIDIA and New York University (NYU).  All material is available under the [Creative Commons Attribution-NonCommercial License](http://creativecommons.org/licenses/by-nc/4.0/).

The labs are designed to be open-ended, multidisciplinary, one- to three-week programming and written assignments for students. Each lab contains a description, sample code, sample solutions and suggestions on how instructors can evaluate and have students submit solutions. 

## System Requirements

#### NVIDIA CUDA/GPUs

Thanks to the rapid development of NVIDIA GPUs, training deep neural networks is more efficient than ever in terms of both time and resource cost. Training neural networks on [NVIDIA CUDA-enabled GPUs](https://developer.nvidia.com/cuda-gpus) is a practical necessity for the Teaching Kit labs, including both convolutional networks (Lab1 and Lab2) and recurrent networks (Lab4).

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

## About the Labs/solutions

#### Recommended prerequisite Teaching Kit lectures for each lab
* Lab1: Module 1 - Introduction to Machine Learning through Module 2 - Introduction to Deep Learning
* Lab2: Module 1 through Module 3 - Convolutional Neural Networks
* Lab3: Module 1 through Module 5 - Optimization Techniques
* Lab4: Module 1 through Module 6 - Learning with Memory

#### Lab documents
`documents` in each lab directory contains the same lab description and sample solution write-up `.pdf` documents in the Teaching Kit `.zip` package downloadable from the [GPU Educators Program portal](https://developer.nvidia.com/educators).

#### Baseline sample code
`sample_code` in each each lab directory contains the baseline training model solution (as well as instructions to run) described in the lab descriptions. These baseline models render a baseline score for the given dataset that students are suggested to outperform. The `sample_code` is designed to be given to students when the lab is assigned.

#### Lab solutions
`solution_code` in each lab directory contains an example implementation of approaches that improve the model performance. These solutions were developed by real students who took the Deep Learning curriculum course at NYU. Some solutions may require additional, undocumented instructions to properly execute. 
Unlike the `sample_code`, the solution samples are not designed to run "out-of-box", but should still provide useful examples of solutions using a variety of techniques for both instructors and students to learn from.
However, the software structure remains the same as `sample_code` and uses the same execution script in the `sample_code` `Readme`s to run. 
Note that for each lab, the sample solution code corresponds to only the 1st "lab*n*_*labName*_solution1.pdf" solution write-up. These solution write-ups are found in both the Teaching Kit `.zip` package downloadable from the [GPU Educators Program portal](https://developer.nvidia.com/educators) and the `documents` folder in each lab directory in this repository.

#### Cloning and Accessing the Labs/solutions

To clone the Labs/solutions on your machine and, for example, access Lab1:
```
    git clone git@bitbucket.org:junbo_jake_zhao/deeplearningkit.git
    cd Lab1
```

#### In-class competition    
Some programming labs include optimizing a neural network training model and suggest students submit solutions to Kaggle using [Kaggle In Class](https://inclass.kaggle.com/) to compare inference accuracy against each other and against the baseline model score from the `sample_code`. Such a competition can encourage students to study the latest public research papers and technical reports to improve their model accuracy on an open-ended problem. Grading model accuracy could simply be based on whether they outperform the baseline, or perhaps based on class rank.

Please read the Kaggle In Class [FAQ](https://www.kaggle.com/wiki/KaggleInClass) for more information on how to set up your course using Kaggle. Using Kaggle is **not** a requirement to make use of the labs. For example, here is one way to evaluate lab solutions without Kaggle:

- Instructor develops (but not release) a testing dataset with the corresponding groundtruth prediction label file
- Students/teams develop models and compare them based on inference accuracy on a validation subset from a given training set (i.e. MNIST)
- Students/teams develop a `result.lua` file that takes in their model file and the dataset, and returns a model prediction in `.csv` format (see details in lab documents)
- Students/teams submit both their most accurate model and `result.lua` scripts
- Instructor executes the `result.lua` for each student/team's submitted model on the unreleased testing dataset
- Compare the model prediction and groudtruth label on the testing set, and obtain the accuracy
- Use the testing accuracy to evaluate/compare students'/teams' model performance