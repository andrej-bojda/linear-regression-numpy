# Linear Regression from Scratch

Gradient descent on synthetic 2D data — **no sklearn, pure NumPy**.

![Loss curve](plots/loss_curve.png)
![Fitted line](plots/fitted_line.png)

## What's implemented

- Synthetic dataset generation (configurable noise, slope, intercept)
- `LinearRegression` class with `fit()` and `predict()` — manual gradient descent
- MSE loss tracked per epoch and plotted
- Fitted line vs ground truth visualization

## Usage

```bash
git clone https://github.com/<you>/linear-regression-numpy
cd linear-regression-numpy
uv sync
uv run python src/train.py
```

## Results

| Hyperparameter | Value |
|---|---|
| Learning rate | 0.01 |
| Epochs | 1000 |
| Final MSE | ~X.XX |

## Stack

Python 3.11 · NumPy · Matplotlib · zero sklearn
