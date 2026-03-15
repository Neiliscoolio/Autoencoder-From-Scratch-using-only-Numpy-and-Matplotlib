import numpy as np
import requests
import os
from autoencoder1 import autoencoder, loss_function, optimizer, save_weights, load_weights

def download_mnist():
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    if not os.path.exists("mnist.npz"):
        print("Downloading MNIST...")
        r = requests.get(url)
        open("mnist.npz", "wb").write(r.content)
    data = np.load("mnist.npz")
    x_train = data["x_train"].reshape(-1, 784) / 255.0
    x_test = data["x_test"].reshape(-1, 784) / 255.0
    return x_train, x_test

def train_autoencoder(data, epochs=100, batch_size=64):
    num_samples = data.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        data_shuffled = data[indices]
        total_loss = 0
        for i in range(0, num_samples, batch_size):
            batch = data_shuffled[i:i+batch_size]
            outputs = autoencoder.forward(batch)
            loss = loss_function.forward(outputs, batch)
            total_loss += loss
            grad = loss_function.backward()
            autoencoder.backward(grad)
            optimizer.update()
        avg_loss = total_loss / (num_samples // batch_size)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")