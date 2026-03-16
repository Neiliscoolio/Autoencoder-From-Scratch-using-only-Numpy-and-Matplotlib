Autoencoder From Scratch

This is my second ever programming project, built entirely from scratch in Python using only NumPy without the use of 
 any machine learning libraries such as  PyTorch or  TensorFlow.
## What is an Autoencoder?

An autoencoder is a neural network that learns to compress data into a compact representation and then reconstruct the original input from that compression. 
It consists of two components: an encoder that progressively reduces the dimensionality of the input, 
and a decoder that attempts to reverse that process and rebuild the original.

For this project the network takes a 784-dimensional input (a flattened 28x28 MNIST digit image) 
and compresses it down to just 64 numbers which is roughly 8% of the original size. 
The decoder then takes those 64 numbers and reconstructs a 784-dimensional output that should be as close to the original as possible. 
The network receives no labels and no guidance on what features to store. 
Through training alone it discovers on its own that things like stroke direction, curvature, and loop structure.

## Implementation

Everything in this project was implemented manually including the forward pass, backpropagation, and weight updates. 
The codebase is split across several files:

- `model.py` contains the primary features: `LinearLayer`, `ReLU`, `Sigmoid`, `Sequential`, and `SGD`
- `loss.py` contains `MeanSquaredError` with backward pass for gradient computation
- `autoencoder.py` defines the encoder, decoder, and full autoencoder architecture along with weight saving and loading
- `train.py` handles data loading and the training loop with batch processing
- `visualize.py` handles reconstruction visualization and latent space interpolation
- `main.py` ties every component together

One  improvement over my previous project is batch training. 
Rather than updating weights after every single sample, the network processes 64 images simultaneously and averages the gradients before updating. 
This makes training significantly faster and produces more stable weight updates.

## Network Architecture
```
Encoder: 784 → 256 → 128 → 64  (ReLU activations)
Decoder: 64 → 128 → 256 → 784  (ReLU hidden, Sigmoid output)
```

The encoder uses ReLU activations throughout. The decoder uses ReLU in its hidden layers but switches to Sigmoid for the final output layer, which constrains the output values between 0 and 1 to match the normalized pixel values of the input images.

## Results

After training for 100 epochs on 60,000 MNIST images, the reconstructions are nearly indistinguishable from the originals. 
More interestingly, interpolating between two encoded digits in the latent space produces smooth morphing sequences through intermediate shapes that have never existed. 


## How to Run
```
pip install numpy matplotlib requests pillow
python main.py
```

The first run trains from scratch and saves the weights. Every subsequent run loads the saved weights and goes straight to visualization.

## Why I Built This

After building an MNIST classifier from scratch, I wanted to understand unsupervised learning and what it means for a network to learn representations without being told what to look for. The classifier always had a correct answer to aim for. The autoencoder has nothing except the instruction to reconstruct what it receives.

The latent space interpolation was the most interesting part. The fact that blending two encoded representations in a 64-dimensional space produces coherent intermediate digits suggests the network has learned a genuinely structured internal model of what digits look like, not just a lookup table.
