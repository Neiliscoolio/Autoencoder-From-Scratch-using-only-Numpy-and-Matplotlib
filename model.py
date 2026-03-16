import numpy as np

#creates layerrs for the autoencoder
class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)* np.sqrt(2.0 / input_size)

        self.bias = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        self.batch_size = inputs.shape[0]
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, gradient_in,):
        self.gradient_weights = np.dot(self.inputs.T, gradient_in) / self.batch_size
        self.gradient_bias = np.sum(gradient_in, axis=0, keepdims=True) / self.batch_size
        return np.dot(gradient_in, self.weights.T)
#activation functions for the hidden layers of the autoencoder
class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)
    
    def backward(self, gradient_in):
        return gradient_in * (self.inputs > 0)

#sigmoid activation function for the output layer of the autoencoder
class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, grad):
        return grad * self.out * (1 - self.out)

#stacks all the layers allowing for a clean loop    
class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self,inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, gradient_in):
        for layer in reversed(self.layers):
            gradient_in = layer.backward(gradient_in)
        return gradient_in
    
#optimizer, stochastic gradient descent
class SGD:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
    
    def update(self):
        for layer in self.layers:
            if hasattr(layer, "gradient_weights"):
                layer.weights -= self.learning_rate * layer.gradient_weights
                layer.bias -= self.learning_rate * layer.gradient_bias
