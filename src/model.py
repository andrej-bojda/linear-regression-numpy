import numpy as np

def predict(X,w,b):
    return (X @ w) + b

def compute_loss(y,y_pred):
    return np.mean((y - y_pred)**2)

if __name__ == "__main__":
    from src.data import generate_data

    X, y, w_true, b_true = generate_data(2.5, 1.0, 100, 0.3, 42)
    w = np.zeros(1)
    y_pred = predict(X, w , 0.0)
    print(compute_loss(y, y_pred))