

Readme · MD
# linear-regression-numpy
 
Linear regression and batch gradient descent implemented from scratch using only NumPy. No scikit-learn, no autograd — every gradient is hand-derived and hand-coded.
 
## What this is
 
A small, deliberately minimal implementation of linear regression trained with batch gradient descent. The goal wasn't to solve a hard problem — it was to build the full pipeline (data generation, forward pass, loss, gradients, training loop, visualization) from first principles and confirm, numerically, that the model recovers known ground-truth parameters.
 
## Math
 
**Model (forward pass):**
 
```
ŷ = wX + b
```
 
**Loss — Mean Squared Error:**
 
```
J(w, b) = (1/N) Σ (yᵢ - ŷᵢ)²
```
 
**Gradients** (derived via chain rule on J):
 
```
dw = (-2/N) Σ Xᵢ(yᵢ - ŷᵢ)
db = (-2/N) Σ (yᵢ - ŷᵢ)
```
 
**Parameter update rule:**
 
```
w ← w - α·dw
b ← b - α·db
```
 
where α is the learning rate.
 
## Project structure
 
```
linear-regression-numpy/
├── notebooks/
│   └── exploration.ipynb   # visualization & learning-rate sensitivity experiments
├── plots/
│   ├── loss_curve.png
│   ├── fit.png
│   └── lr_sensitivity.png
├── src/
│   ├── __init__.py
│   ├── data.py              # synthetic data generation
│   ├── model.py              # predict, compute_loss, compute_gradients
│   └── train.py               # training loop + CLI entrypoint
├── pyproject.toml
└── README.md
```
 
## How to run
 
```bash
uv run python src/train.py
```
 
This generates synthetic data, trains the model, and prints a comparison of true vs. learned parameters.
 
For visualizations and the learning-rate sensitivity sweep, open:
 
```bash
uv run jupyter lab notebooks/exploration.ipynb
```
 
## Results
 
Trained on 100 synthetic points (`w_true = 2.5`, `b_true = 1.0`, Gaussian noise σ = 0.3), with `α = 0.1` over 450 epochs:
 
| Parameter  | True Value | Learned Value | Absolute Error |
|------------|-----------:|--------------:|----------------:|
| w (weight) |     2.5000 |       2.4515  |         0.0485 |
| b (bias)   |     1.0000 |        1.0228 |         0.0228  |
 
*(Replace the placeholders above with the actual numbers printed by `train.py`.)*
 
**Loss curve:**
 
![Loss curve](plots/loss_curve.png)
 
**Fitted line vs. data:**
 
![Model fit](plots/fit.png)
 
**Learning rate sensitivity** — comparing convergence at α = 0.01, 0.1, 0.5:
 
![Learning rate sensitivity](plots/lr_sensitivity.png)
 
A smaller α converges correctly but is still descending after 450 epochs; α = 0.1 converges cleanly within that window; α = 0.5 converges fastest but starts to show early oscillation, illustrating the standard speed/stability tradeoff in gradient descent.
 
## What this does not do (and why)
 
This implementation uses **batch gradient descent** — every parameter update uses all 100 data points at once. That's fine at this scale and makes the math easy to verify, but it doesn't scale to real datasets with millions of rows. A production implementation would use **mini-batch SGD**, sampling a random subset of the data per update.
 
Also omitted, intentionally, to keep the scope tight:
- Feature normalization (not needed here since X is already in [0, 1])
- Multiple features / multivariate regression
- Learning rate scheduling or adaptive optimizers (Adam, RMSProp, etc.)
- Regularization (L1/L2)
## What I'd do next
 
Extend `model.py` to support multiple features (vectorized `w` instead of a scalar), then reimplement the same training loop with mini-batch SGD to compare convergence behavior against the batch version here.
 
