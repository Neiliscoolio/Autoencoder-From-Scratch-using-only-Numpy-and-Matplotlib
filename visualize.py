from autoencoder1 import autoencoder, encoder, decoder
import matplotlib.pyplot as plt
import numpy as np


def visualize(x_test, n=10):
    fig, axes = plt.subplots(2, n, figsize=(15, 3))
    fig.patch.set_facecolor('black')
    for i in range(n):
        img = x_test[i:i+1]
        reconstructed = autoencoder.forward(img)
        axes[0, i].imshow(img.reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed.reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_title('Original', color='white', fontsize=10)
    axes[1, 0].set_title('Reconstructed', color='white', fontsize=10)
    plt.tight_layout()
    plt.show()

def interpolate(x_test):
    img_a = x_test[0:1]
    img_b = x_test[1:2]
    code_a = encoder.forward(img_a)
    code_b = encoder.forward(img_b)
    t_values = np.linspace(0, 1, 10)
    fig, axes = plt.subplots(1, len(t_values), figsize=(15, 2))
    fig.patch.set_facecolor('black')
    for i, t in enumerate(t_values):
        code = code_a * (1 - t) + code_b * t
        recon = decoder.forward(code)
        axes[i].imshow(recon.reshape(28, 28), cmap='gray')
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()
