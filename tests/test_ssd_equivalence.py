"""
Numerical proof (NumPy only) that the matrix-form bidirectional SSD mixer used in
MicroPathMamba equals a forward + backward selective-scan recurrence.

Forward scan  : h_t = a_t h_{t-1} + dt_t B_t x_t^T,          y_t = C_t^T h_t
Backward scan : g_t = a_{t+1} g_{t+1} + dt_t B_t x_t^T,      u_t = C_t^T g_t
with a_t = exp(dt_t * A), A < 0. Bidirectional output = y + u - (diagonal term counted twice).

Matrix form   : Y = (exp(-|c_i - c_j|) * (C B^T)) @ (dt * X),  c = cumsum(dt * A)
cumsum is computed as a MatMul with a constant triangular matrix, so the whole block
is MatMul / Mul / Sub / Abs / Exp only - operators an NPU toolchain can map statically.
Run: python tests/test_ssd_equivalence.py
"""
import numpy as np

rng = np.random.default_rng(0)
L, N, P = 49, 16, 8          # tokens (7x7), state size, head dim
A = -np.exp(rng.normal(size=()))           # scalar decay per head (Mamba-2 style)
dt = 0.5 / (1 + np.exp(-rng.normal(size=L)))   # bounded sigmoid step, as in the model
B = rng.normal(size=(L, N))
C = rng.normal(size=(L, N))
X = rng.normal(size=(L, P))
a = np.exp(dt * A)

# --- sequential reference -----------------------------------------------------
h = np.zeros((N, P)); y_f = np.zeros((L, P))
for t in range(L):
    h = a[t] * h + dt[t] * np.outer(B[t], X[t])
    y_f[t] = C[t] @ h
g = np.zeros((N, P)); y_b = np.zeros((L, P))
for t in reversed(range(L)):
    g = (a[t + 1] * g if t + 1 < L else g) + dt[t] * np.outer(B[t], X[t])
    y_b[t] = C[t] @ g
diag = (np.sum(C * B, axis=1) * dt)[:, None] * X
y_seq = y_f + y_b - diag

# --- matrix (NPU-friendly) form ----------------------------------------------
tri = np.triu(np.ones((L, L)))          # tri[k, i] = 1 if k <= i
c = (dt * A) @ tri                       # cumsum via MatMul
M = np.exp(-np.abs(c[:, None] - c[None, :]))
y_mat = (M * (C @ B.T)) @ (dt[:, None] * X)

err = np.max(np.abs(y_seq - y_mat)) / np.max(np.abs(y_seq))
print(f"max relative error = {err:.2e}")
assert err < 1e-10, err

# causal (forward-only) special case
y_causal = (np.tril(np.exp(c[:, None] - c[None, :])) * (C @ B.T)) @ (dt[:, None] * X)
assert np.max(np.abs(y_causal - y_f)) / np.max(np.abs(y_f)) < 1e-10
print("PASS: bidirectional and causal matrix forms match the sequential scan")
