import numpy as np
import matplotlib.pyplot as plt

def generate_data(w_true, b_true, n_samples = 100, noise_scale= 0.1, seed = 42):
    np.random.seed(seed)
    X = np.random.rand(n_samples, 1)
    noise = noise_scale * np.random.randn(n_samples, 1)
    y = (w_true * X) + b_true + noise
    return X,y, w_true, b_true

if __name__ == "__main__" :
    X, y, _, _ = generate_data(w_true=2, b_true=1)
    plt.scatter(X, y, color='blue', alpha=0.7)
    plt.show()