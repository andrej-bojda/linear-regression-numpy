from model import predict, compute_loss, compute_gradients

def train(X, y, alpha, epochs):
    w = 0.0
    b = 0.0
    loss_history = []
    for epoch in range(epochs):
        y_pred = predict(X, w, b)
        loss_history.append(compute_loss(y, y_pred))
        dw, db = compute_gradients(X, y, y_pred)
        w = w - (alpha * dw)
        b = b - (alpha * db)
    return w, b, loss_history

if __name__ == "__main__":
    from data import generate_data
    X, y, w_true, b_true = generate_data(w_true = 2.5, b_true = 1.0)
    w, b, loss_history = train(X, y, alpha = 0.1, epochs = 2000)
    w_error = abs(w_true - w)
    b_error = abs(b_true - b)
    print("\n" + "=" * 55)
    print(f"{'Parameter':<10} | {'True Value':<10} | {'Learned Value':<13} | {'Absolute Error':<10}")
    print("=" * 55)
    print(f"{'w (weight)':<10} | {w_true:<10.4f} | {w:<13.4f} | {w_error:<10.4f}")
    print(f"{'b (bias)':<10} | {b_true:<10.4f} | {b:<13.4f} | {b_error:<10.4f}")
    print("=" * 55)
    print(f"Final Loss after {len(loss_history)} epochs: {loss_history[-1]:.6f}\n")
    print("First loss:", loss_history[0])
    print("Last loss:", loss_history[-1])
