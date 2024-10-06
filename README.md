# Sparse Autoencoder for Steering Mistral 7B

This repository contains a Sparse Autoencoder (SAE) designed to interpret and steer the Mistral 7B language model. By training the SAE on the residual activations of Mistral 7B, we aim to understand the internal representations of the model and manipulate its outputs in a controlled manner.

## Overview

Large Language Models (LLMs) like Mistral 7B have complex internal mechanisms that are not easily interpretable. This project leverages a Sparse Autoencoder to:

- **Decode internal activations**: Transforming high-dimensional activations into sparse, interpretable features.
- **Steer model behavior**: Manipulating specific features to influence the model's output.

This approach is based on the hypothesis that internal features are superimposed in the model's activations and can be disentangled using sparse representations.

## Personal Work

I have written the following articles that provide foundational insights guiding the development of this project:

- [**Reversing Transformer to Understand In-Context Learning with Phase Change, Feature Dimensionality, and Gradient Descent**](https://seongland.medium.com/reversing-transformer-to-understand-in-context-learning-with-phase-change-feature-dimensionality-13cbf8a2f984)
    
    This article explores how reversing transformers can shed light on in-context learning mechanisms, phase transitions, and feature dimensionality in large language models.
    
- [**Superposition Hypothesis for Steering LLM with Sparse Autoencoder**](https://seongland.medium.com/superposition-hypothesis-for-steering-llm-with-sparse-autoencoder-c07b74d23e96)
    
    This post discusses how the superposition hypothesis can be applied to steer large language models using sparse autoencoders by isolating and manipulating specific features within the model.
    

These writings provide foundational insights that have guided the development of this project.

## Installation

1. **Clone the repository:**
    
    ```bash
    git clone https://github.com/yourusername/mistral-sae.git
    cd mistral-sae
    ```
    
2. **Install dependencies:**
    
    ```bash
    pip install -r requirements.txt
    ```
    
    Ensure you have the appropriate version of PyTorch installed, preferably with CUDA support for GPU acceleration.
    

## Usage

### Training the Sparse Autoencoder

The `train.py` script trains the SAE on activations from a specified layer of the Mistral 7B model.

```bash
python train.py
```

- Adjust hyperparameters like `D_MODEL`, `D_HIDDEN`, `BATCH_SIZE`, and `lr` within the script.
- Set the `MISTRAL_MODEL_PATH` and `target_layer` to specify which model and layer to use.

### Generating Feature Explanations

Use `explain.py` to generate natural language explanations for the features learned by the SAE.

```bash
python explain.py
```

- Ensure you have access to the required datasets (e.g., The Pile) and APIs.
- Configure parameters such as `batch_size`, `data_path`, and `target_layer`.

### Steering the Model Output

The `demo.py` script demonstrates how to steer the Mistral 7B model by manipulating specific features.

```bash
python demo.py
```

- Set `FEATURE_INDEX` to the index of the feature you wish to manipulate.
- Toggle `STEERING_ON` to `True` to enable steering.
- Adjust the `coeff` variable to control the strength of the manipulation.

## Project Structure

- `config.py`: Contains model configurations and helper functions.
- `train.py`: Script for training the Sparse Autoencoder.
- `explain.py`: Generates explanations for the features identified by the SAE.
- `demo.py`: Demonstrates how to steer the Mistral 7B model using the SAE.
- `mistral_sae/`: Directory containing the SAE implementation and related utilities.
- `requirements.txt`: Lists the Python dependencies required for the project.

## Background

Understanding the internal workings of LLMs is crucial for both interpretability and control. By applying a Sparse Autoencoder to the activations of Mistral 7B, we can:

- Identify **monosemantic neurons** that correspond to specific concepts or features.
- Test the **superposition hypothesis** by examining how multiple features are represented within the same neurons.
- Enhance our ability to **steer the model's outputs** towards desired behaviors by manipulating these features.

## Acknowledgments

This project is inspired by and builds upon several key works:

- [**Scaling Monosemanticity: Extracting Interpretable Features from Claude 3**](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
    
    *Templeton et al., 2024*
    
    Explores methods for extracting interpretable features from large language models.
    
- [**Scaling and Evaluating Sparse Autoencoders**](https://arxiv.org/abs/2406.04093v1)
    
    *Gao et al., 2024*
    
    Discusses techniques for scaling sparse autoencoders and evaluating their performance.


## Resources

- **Model Weights**: Available on Hugging Face at [tylercosgrove/mistral-7b-sparse-autoencoder-layer16](https://huggingface.co/tylercosgrove/mistral-7b-sparse-autoencoder-layer16).
- **SAE Lens Compatibility**: This project is compatible with [SAE Lens](https://github.com/jbloomAus/SAELens), a tool for analyzing sparse autoencoders.

##