# Bernstein Paper Experiments

This repo contains the TensorFlow experiments of our paper

## Getting Started

### Install Prerequisites

 1. Bernstein Flow
    ```bash
    pip install git+https://github.com/MArpogaus/TensorFlow-Probability-Bernstein-Polynomial-Bijector.git
    ```
 2. TensorFlow Experiments
    ```bash
    pip install git+https://github.com/MArpogaus/tensorflow-experiments.git
    ```

### Clone and install this repo

Or clone this repository an install it from there:

```bash
git clone https://github.com/MArpogaus/Bernstein-Experiments.git ./exp
cd exp
pip install -e .
```

## Usage

Either use my tfexp cli command to run the training of a given configuration, i.e.:

```bash
tfexp train configs/feed_forward_bernstein_flow.yaml
```

or use the provided Notebooks `tfexp_train.ipynb` for training and `tfexp_evaluate.ipynb` to evaluate a model.
