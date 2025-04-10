# RL-Transformer-GAN: Reinforcement Learning Enhanced Transformer GAN for Trajectory Generation

This project implements a novel approach to trajectory generation using a combination of Reinforcement Learning (RL), Transformer architecture, and Generative Adversarial Networks (GAN). The model is designed to generate realistic and diverse trajectories while maintaining the underlying patterns and characteristics of real-world movement data.

## Overview

The RL-Transformer-GAN combines three powerful deep learning paradigms:
- **Transformer Architecture**: For capturing long-range dependencies and complex patterns in trajectory data
- **Generative Adversarial Networks**: For generating realistic trajectories
- **Reinforcement Learning**: For optimizing the generation process through reward signals

## Project Structure

```
.
├── model.py              # Core model implementation
├── train.py             # Training script
├── inference.py         # Inference and generation script
├── losses.py            # Custom loss functions
├── test_utility.py      # Testing utilities
├── TUL_test.py          # Trajectory Utility Learning tests
├── data/                # Data directory
├── params/              # Model parameters
├── training_params/     # Training configurations
├── results/             # Training results and checkpoints
└── MARC/                # MARC-related utilities
```

## Features

- **Transformer-based Architecture**: Utilizes multi-head attention mechanisms for better sequence modeling
- **RL-Enhanced Training**: Incorporates reinforcement learning to optimize trajectory generation
- **Custom Loss Functions**: Implements specialized loss functions for trajectory generation
- **Flexible Generation**: Supports both conditional and unconditional trajectory generation
- **Checkpoint Management**: Built-in checkpoint saving and loading functionality

## Requirements

- TensorFlow 2.x
- Keras
- NumPy
- Other dependencies (see requirements.txt)

## Usage

### Training

```bash
python train.py
```

### Inference (test data generater)

```bash
python inference.py
```

### Evaluation

The project provides two evaluation methods:

1. **Basic Test Utility**
```bash
python test_utility.py
```

2. **Trajectory Utility Learning (TUL) Test**
```bash
python test.py data/train_latlon.csv results/syn_traj_test.csv 32
```
This command evaluates the generated trajectories against the training data, where:
- `data/train_latlon.csv`: Path to the training data
- `results/syn_traj_test.csv`: Path to the generated trajectories
- `32`: Embedding

## Model Architecture

The model consists of three main components:
1. **Generator**: Transformer-based network that generates trajectories
2. **Discriminator**: Evaluates the authenticity of generated trajectories
3. **Critic**: Provides value estimates for RL optimization

## Training Process

The training process combines:
- GAN training with custom loss functions
- RL optimization using advantage estimation
- Transformer-based sequence modeling

## Results

The model generates trajectories that:
- Maintain realistic movement patterns
- Preserve spatial and temporal characteristics
- Show diversity in generated samples
- Respect geographical constraints

