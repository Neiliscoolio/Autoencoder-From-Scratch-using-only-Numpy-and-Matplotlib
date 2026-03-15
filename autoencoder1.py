from model import SGD, Sequential, LinearLayer, ReLU, Sigmoid
from loss import MeanSquaredError
import numpy as np

encoder = Sequential([
    LinearLayer(784, 256),
    ReLU(),
    LinearLayer(256, 128),
    ReLU(),
    LinearLayer(128, 64),
])

decoder = Sequential([
    LinearLayer(64, 128),
    ReLU(),
    LinearLayer(128, 256),
    ReLU(),
    LinearLayer(256, 784),
    Sigmoid()
])
autoencoder = Sequential(encoder.layers + decoder.layers)

# loss and optimizer
loss_function = MeanSquaredError()
optimizer = SGD(autoencoder.layers, learning_rate=0.01)
def save_weights(path='ae_weights.npz'):
    np.savez(path,
        w0=autoencoder.layers[0].weights, b0=autoencoder.layers[0].bias,
        w1=autoencoder.layers[2].weights, b1=autoencoder.layers[2].bias,
        w2=autoencoder.layers[4].weights, b2=autoencoder.layers[4].bias,
        w3=autoencoder.layers[5].weights, b3=autoencoder.layers[5].bias,
        w4=autoencoder.layers[7].weights, b4=autoencoder.layers[7].bias,
        w5=autoencoder.layers[9].weights, b5=autoencoder.layers[9].bias,
    )

def load_weights(path='ae_weights.npz'):
    data = np.load(path)
    autoencoder.layers[0].weights = data['w0']
    autoencoder.layers[0].bias = data['b0']
    autoencoder.layers[2].weights = data['w1']
    autoencoder.layers[2].bias = data['b1']
    autoencoder.layers[4].weights = data['w2']
    autoencoder.layers[4].bias = data['b2']
    autoencoder.layers[5].weights = data['w3']
    autoencoder.layers[5].bias = data['b3']
    autoencoder.layers[7].weights = data['w4']
    autoencoder.layers[7].bias = data['b4']
    autoencoder.layers[9].weights = data['w5']
    autoencoder.layers[9].bias = data['b5']
