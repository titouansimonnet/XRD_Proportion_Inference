# Deep neural network method for mineral phases quantification using XRD patterns

## Abstract
We propose a method to quantify the mineral phases of a material sample using the X-Ray diffraction patterns. 
This is a two steps method. 
1. We first train a Neural Network (NN) using synthetic data. 
1. Then, recovering the trained NN we are able to test the method with 32 experimental data. 

![Abstract](./Figures/Abstract.svg)

## Prerequisites

- python 3.9.7
- pytorch 1.10.1
- [crystals](https://crystals.readthedocs.io/en/master/index.html) 1.4.0
- [PyCifRW](https://pypi.org/project/PyCifRW/4.1/) 4.4.6

## Getting started

### 1 - Clone repository 
```
git clone https://github.com/titouansimonnet/XRD_Proportion_Inference

cd XRD_Proportion_Inference
```
### 2 -  Set environnement
We use a local conda virtual environnement 

```
conda create -n XRD
source activate XRD 
```

Install dependencies

```
conda install python=3.9.7
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
conda install -c conda-forge crystall=1.4.0
conda install -c anaconda pandas=1.3.4
```

Alternately, it is possible to create the virtual environnement on a remote server

```
python3 -m vend XRD
```

## Use the code
### 1 - Generate synthetic database


```
cd Simulation_XRD_patterns
python Mineral_phase_generation.py
```
