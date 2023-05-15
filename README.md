# Master Thesis - Transfer learning with convolutional neural networks for myocardial infarction classification in ECGs 

This repository includes a PyTorch implementation of the master thesis 'Transfer learning with convolutional neural networks for myocardial infarction classification in ECGs'. A link to the accompanying paper will be supplied, as soon, as it is released.

## Conda Environment
A working conda environment can be created using the supplied environment.yaml file

## Network Architecures
The base models are called GeneralistNeuralNet
The specialist models are called SpecialistNeuralNet
The Upscaling models are called UpscalingNeuralNet

It is assumed, that first the base model is trained, with the specialist follwing second, and the upscaling model third.