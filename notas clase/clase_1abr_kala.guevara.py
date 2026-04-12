# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:12:48 2026

@author: gueva
"""

import numpy as np
import matplotlib.pyplot as plt

#dft
N = 8
n = np.arange(N)
x1 = 4 + 3 * np.sin(np.pi/2 * n)
X1 = np.fft.fft(x1)
X1_mod = np.abs(X1)
X1_ph = np.angle(X1)

#grafiquitos
plt.figure()
plt.stem(n, X1_mod, basefmt=" ")
plt.axhline(0, color='black', linewidth=1)
plt.title("Modulo DFT (π/2)")
plt.xlabel("k")
plt.ylabel("|X[k]|")
plt.grid()
plt.show()

plt.figure()
plt.stem(n, X1_ph, basefmt=" ")
plt.axhline(0, color='black', linewidth=1)
plt.title("Fase DFT (π/2)")
plt.xlabel("k")
plt.ylabel("Fase")
plt.grid()
plt.show()

#potencia espectral
Potencia1 = (X1_mod**2) / (N**2)

plt.figure()
plt.stem(n, Potencia1, basefmt=" ")
plt.axhline(0, color='black', linewidth=1)
plt.title("Espectro de potencia (π/2)")
plt.xlabel("k")
plt.ylabel("Potencia")
plt.grid()
plt.show()

#%% probamos con otra frec
x2 = 4 + 3 * np.sin(3 * np.pi/2 * n)
X2 = np.fft.fft(x2)
X2_mod = np.abs(X2)
X2_ph = np.angle(X2)

plt.figure()
plt.stem(n, X2_mod, basefmt=" ")
plt.axhline(0, color='black', linewidth=1)
plt.title("Modulo DFT (3π/2)")
plt.xlabel("k")
plt.ylabel("|X[k]|")
plt.grid()
plt.show()

plt.figure()
plt.stem(n, X2_ph, basefmt=" ")
plt.axhline(0, color='black', linewidth=1)
plt.title("Fase DFT (3π/2)")
plt.xlabel("k")
plt.ylabel("Fase")
plt.grid()
plt.show()

#potencia espectral
Potencia2 = (X2_mod**2) / (N**2)

plt.figure()
plt.stem(n, Potencia2, basefmt=" ")
plt.axhline(0, color='black', linewidth=1)
plt.title("Espectro de potencia (3π/2)")
plt.xlabel("k")
plt.ylabel("Potencia")
plt.grid()
plt.show()


