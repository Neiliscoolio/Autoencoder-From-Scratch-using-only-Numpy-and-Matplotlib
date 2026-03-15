import os
from training import download_mnist, train_autoencoder
from visualize import visualize, interpolate
from autoencoder1 import save_weights, load_weights

if __name__ == "__main__":
    x_train, x_test = download_mnist()
    if os.path.exists('ae_weights.npz'):
        print("Loading saved weights...")
        load_weights()
    else:
        print("Training...")
        train_autoencoder(x_train, epochs=100, batch_size=64)
        save_weights()
        print("Weights saved!")
    visualize(x_test)
    interpolate(x_test)