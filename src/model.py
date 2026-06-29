import numpy as np

def predict(X,w,b):
    return (X * w) + b

def compute_loss(y,y_pred):
    return np.mean((y - y_pred)**2)

def compute_gradients(X, y, y_pred):
    return np.mean(-2 * (y - y_pred) * X), np.mean(-2 * (y - y_pred))

if __name__ == "__main__":
    from src.data import generate_data
    X, y, w_true, b_true = generate_data(2.5, 1.0, 100, 0.3, 42)
    w = np.zeros((1,1))
    b = 0.0
    y_pred = predict(X, w , 0.0)
    dw, db = compute_gradients(X, y, y_pred)
    print(f"dw : {dw}")
    print(f"db : {db}")


