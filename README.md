# Transformer: "Attention is All You Need" Implementation in PyTorch

This repository contains an implementation of the Transformer model as described in the seminal paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). The Transformer model is a neural network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. This implementation is done using PyTorch, and utilizes Hugging Face's tokenizer for text preprocessing.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)

## Introduction

The Transformer model introduced a new mechanism called "self-attention" which allows the model to focus on different parts of the input sequence as it processes it. This has significantly improved performance in many natural language processing tasks.

## Installation

To install the dependencies required for this project, use the following command:

```bash
git clone https://github.com/eryash15/transformer_pytorch.git
cd transformer_pytorch
```


```bash
pip install -r requirements.txt
```

## Model Architecture

The Transformer model consists of an encoder and a decoder. Both the encoder and decoder are composed of a stack of identical layers. Each layer has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network.

#### Encoder

The encoder is composed of a stack of N identical layers. Each layer has two sub-layers:

Multi-head self-attention mechanism
Position-wise fully connected feed-forward network

#### Decoder

The decoder is also composed of a stack of N identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.

## Training

To train the Transformer model, use the provided notebook experiments/tokenizer_dataset_train.ipynb. Make sure to configure the training parameters in the config.json file.

## Evaluation

To train the Transformer model, use the provided notebook experiments/evaluation.ipynb script.

## Contributing

We welcome contributions to this project! If you find a bug or have a feature request, please open an issue. If you would like to contribute code, please fork the repository and submit a pull request. 
