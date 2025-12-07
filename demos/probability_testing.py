import numpy as np
import math
import matplotlib.pyplot as plt

# mL parameter (typically mL = 1 / ln(M))
mL = 1.0

def sample_closed_form(mL):
    U = np.random.random()
    return int(-math.log(U) * mL)

def sample_loop(mL):
    L = 0
    p = math.exp(-1.0/mL)
    while np.random.random() < p:
        L += 1
    return L

# Generate samples
N = 200000
closed_samples = np.array([sample_closed_form(mL) for _ in range(N)])
loop_samples   = np.array([sample_loop(mL) for _ in range(N)])

# Plot them
plt.figure(figsize=(10,5))
bins = np.arange(0,20)

plt.hist(closed_samples, bins=bins, alpha=0.5, label='Closed-form', density=True)
plt.hist(loop_samples,   bins=bins, alpha=0.5, label='While-loop', density=True)

plt.title("Comparison of level distributions")
plt.xlabel("Level L")
plt.ylabel("Probability")
plt.legend()
plt.show()
