import numpy as np
import matplotlib.pyplot as plt

# Frequency axis
f = np.linspace(-10, 10, 4000)

# Triangular spectrum
def triangular(f, B):
    return np.maximum(1 - np.abs(f)/B, 0)

B = 0.8  # narrow to keep peaks separated

def build_replicas(fs):
    X = np.zeros_like(f)
    for k in range(-5, 6):
        X += triangular(f - k*fs, B)
    return X

# Case 1: fs = 2
X1 = build_replicas(2)
# Case 2: fs = 4
X2 = build_replicas(4)

plt.figure()
plt.plot(f, X1)
plt.title("Réplicas espectrales - fs = 2 Hz")
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.grid()
plt.show()

# Case 2: fs = 4
X2 = build_replicas(4)

plt.figure()
plt.plot(f, X2)
plt.title("Réplicas espectrales - fs = 4 Hz")
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.grid()
plt.show()
