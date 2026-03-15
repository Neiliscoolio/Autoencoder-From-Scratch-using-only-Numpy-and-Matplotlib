import numpy as np

class MeanSquaredError:
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        return np.mean((predictions - targets)**2)
    
    def backward(self):
        batch_size = self.targets.shape[0]
        return 2 * (self.predictions - self.targets) / batch_size