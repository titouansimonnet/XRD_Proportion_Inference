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
- [PyCifRW](https://pypi.org/project/PyCifRW/4.1/) 4.4.3

## 1- Generation of synthetic XRD patterns
