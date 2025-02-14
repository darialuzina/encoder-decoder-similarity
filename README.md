# Similarity Computation in Neural Networks

This repository contains two implementations of similarity computation between encoder and decoder context vectors using PyTorch.

## Overview

Similarity computation is essential in sequence-to-sequence models, particularly in tasks such as machine translation, text generation, and attention mechanisms. This repository provides two approaches:

- **Similarity1**: Computes similarity using a simple dot product between encoder states and the decoder state.
- **Similarity2**: Implements a more advanced similarity function using linear transformations and a nonlinear activation function.

## Similarity Functions

### **Similarity1**
The similarity function in `Similarity1` is computed using the **dot product**:

$$
\text{sim}(h, s) = h^T s = \sum_{i} h_i s_i
$$

where:
- \( h \) represents the encoder states.
- \( s \) is the decoder state.
- The dot product measures alignment between these vectors.



### **Similarity2**
The similarity function in `Similarity2` is defined as:

\[
\text{sim}(h, s) = \text{fc}_3 (\tanh(\text{fc}_1(h) + \text{fc}_2(s)))
\]

where:
- \( \text{fc}_1 \) and \( \text{fc}_2 \) are linear transformations that project the encoder and decoder states into an intermediate representation.
- The **tanh** activation introduces non-linearity.
- \( \text{fc}_3 \) maps the combined representation to a final similarity score.

This approach allows for more flexibility in learning complex similarity patterns.

## Applications

### **Neural Machine Translation**
These similarity functions can be used in sequence-to-sequence architectures to measure how well an encoder state aligns with the decoder’s current state.

### **Attention Mechanisms**
Both similarity methods can be integrated into attention mechanisms to assign importance weights to different encoder outputs, helping the decoder focus on relevant information.

### **Information Retrieval**
Used in search engines and recommendation systems to compare query vectors with stored document embeddings, ranking them by relevance.

### **Text Generation**
These similarity models can help select contextually appropriate encoder states in tasks like chatbot response generation and text summarization.

### **Ranking Systems**
In recommendation engines and ranking models, similarity scores help determine how closely two representations match, improving personalization and retrieval accuracy.

## Files

- **`similarity1.py`**: Implements a basic dot product similarity function between encoder and decoder states.
- **`similarity2.py`**: Uses fully connected layers and a nonlinear activation function (`Tanh`) to compute similarity in a transformed space.
