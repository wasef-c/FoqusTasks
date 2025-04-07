# FoqusTasks

## Overview

This code demonstrates the generation of synthetic MRI data and trains 5 different model types on the regeneration of both undersampled and fully sampled slices.

## Generating The Dataset

`generate_dataset.py` contains the functions to generate the synthetic data. To save the generated data into a folder, run `generate_sample_data.py`.

## Training the 5 Models

`train.py` will run the training of five different models:
- Spatial data only
- Frequency Data only
- Parallel Hybrid
- Hybrid Sequential
- Hybrid Sequential using intermediate skip connections

For more details on these models, their implementations can be found in `model_functions.py`. The Hybrid Sequential model is inspired by the paper "A Hybrid Frequency-domain/Image-domain Deep Network for Magnetic Resonance Image Reconstruction" by Roberto Souza and Richard Frayne.

Additionally, `Experimenting.ipynb` shows a sample output of the training process, showing the loss curves and outputs

## Evaluating the Model on Noisy, Undersampled Data and Random Data Loss

`evaluate.py` is set to evaluate the Hybrid Sequential Model (without skip connections), as it performed best on the test set from `train.py`. This script generates the folders `evaluation_examples` and `evaluation_results`. You will need a trained model from `train.py` in the parent directory for this to work.

The results show that the model does not perform well on noisy and 2x acceleration data. To mitigate this, I suggest adding transforms to the dataloader including rotation, random noise addition, and random undersampling.

## Inference

`inference.py` will run an inference on a folder of undersampled data. This script has default arguments for input data directory, output directory, model path and model type. It defaults to the Hybrid Sequential Model. Examples of these inference results can be found in the folder `InferenceResults_HybridSequential`.
