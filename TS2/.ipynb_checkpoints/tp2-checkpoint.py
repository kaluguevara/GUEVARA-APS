# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 12:08:42 2025

@author: gueva
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square

# ================================================================
# PUNTO 1
# ================================================================

# las señales de entrada que generó en el TS1
frec = 2000  # Hz
frec_muestreo = 50000  # Hz
Ts = 1/frec_muestreo

t_seno = np.arange(0, 0.005, Ts)       
t_cuadrada = np.arange(0, 0.005, Ts)   
t_pulso = np.arange(0, 0.02, Ts)        

# Señales de entrada
x1 = np.sin(2 * np.pi * frec * t_seno)
x2 = 2 * np.sin(2 * np.pi * frec * t_seno + np.pi/2)

frecmoduladora = 1000  
m = 0.6
x3 = (1 + m * np.sin(2 * np.pi * frecmoduladora * t_seno)) * x1

A = np.max(np.abs(x1))   
Aclip = 0.75 * A        
x1_clippeada = np.clip(x1, -Aclip, Aclip)

frec_cuadrada = 4000
amp = 1    
x_cuadrada = amp * square(2 * np.pi * frec_cuadrada * t_cuadrada)

pulso = 0.01
x_pulso = np.where(t_pulso < pulso, 1, 0)

# ecuación en diferencias que modela un sistema LTI, considerando que son causales
def sistema_LTI(x):
    N = len(x)
    y = np.zeros(N)
    for n in range(N):
        # entrada
        xn   = 0.03*x[n]
        xn1  = 0.05*x[n-1] if n-1 >= 0 else 0
        xn2  = 0.03*x[n-2] if n-2 >= 0 else 0
        # salida
        yn1  = 1.5*y[n-1] if n-1 >= 0 else 0
        yn2  = -0.5*y[n-2] if n-2 >= 0 else 0
        y[n] = xn + xn1 + xn2 + yn1 + yn2
    return y

# calculo salidas
y1 = sistema_LTI(x1)
y2 = sistema_LTI(x2)
y3 = sistema_LTI(x3)
y1_clip = sistema_LTI(x1_clippeada)
y_sq = sistema_LTI(x_cuadrada)
y_pulso = sistema_LTI(x_pulso)

# grafico
fig, axs = plt.subplots(6, 2, figsize=(12, 12))

signals = [
    ("x1 seno 2kHz", t_seno, x1, y1),
    ("x2 seno 2kHz amplif/fase", t_seno, x2, y2),
    ("x3 AM", t_seno, x3, y3),
    ("x1 clippeada", t_seno, x1_clippeada, y1_clip),
    ("x_cuadrada 4kHz", t_cuadrada, x_cuadrada, y_sq),
    ("x_pulso 10ms", t_pulso, x_pulso, y_pulso),
]

for i, (name, t, x, y) in enumerate(signals):
    axs[i,0].plot(t, x)
    axs[i,0].set_title(f"Entrada {name}")
    axs[i,0].grid(True)
    axs[i,1].plot(t, y)
    axs[i,1].set_title(f"Salida {name}")
    axs[i,1].grid(True)

plt.tight_layout()
plt.show()

'''
me piden "En cada caso indique la frecuencia de muestreo, 
el tiempo de simulación y la potencia o energía de la señal de salida."
'''
def calc_potencia(x): 
    return np.mean(x**2)

def calc_energia(x, Ts): 
    return np.sum(x**2) * Ts

def printear_datos():
    print("Frecuencia de muestreo, tiempo de simulación y potencia/energía de la salida:")
    for nombre, t, sig, y in signals:
        tiempo_sim = t[-1] - t[0]
        if "pulso" in nombre:  # energía si es de duración finita
            E = calc_energia(y, Ts)
            print(f"{nombre}: fs = {frec_muestreo} Hz, Tsim = {tiempo_sim:.5f} s, Energía = {E:.5f}")
        else:                  # potencia si es periódica
            P = calc_potencia(y)
            print(f"{nombre}: fs = {frec_muestreo} Hz, Tsim = {tiempo_sim:.5f} s, Potencia = {P:.5f}")
#t[0] = instante inicial (≈ 0), t[-1] = último instante simulado.
#el intervalo total de tiempo de la simulación es la diferencia

print("Datos primer bulletpoint:\n")
printear_datos()
print("\n")

# Hallar la respuesta al impulso 
N = 50
delta = np.zeros(N)
delta[0] = 1
h = sistema_LTI(delta)

# convolucion manual
def convolucion(u):
    y = np.zeros(len(u))
    for n in range(len(u)):
        suma = 0
        for k in range(len(h)):
            if n-k >= 0:
                suma += h[k] * u[n-k]
        y[n] = suma
    return y

#convoluciono tutti y las grafioc
for nombre, t, sig, y in signals:
    # Salidas
    y1_directa = sistema_LTI(sig)
    y1_conv = convolucion(sig)[:len(sig)]   # recorto para que tenga la misma longitud
    
    # Gráfico comparativo
    plt.figure(figsize=(10,5))
    plt.plot(t, y1_directa, label="Salida directa (sistema)", color="b")
    plt.plot(t, y1_conv, "--", label="Salida por convolución", color="r")
    plt.title(f"Comparación de salidas para {nombre}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid(True)
    plt.show()
    
print("Datos segundo bulletpoint:\n")
printear_datos()

# ================================================================
# PUNTO 2
# ================================================================

# rta al impulso y la salida correspondiente a una señal de entrada senoidal 
#en los sistemas definidos mediante las siguientes ecuaciones en diferencias:

def sistema1(x):
    N = len(x)
    y = np.zeros(N)
    for n in range(N):
        xn   = x[n]
        xn10 = 3*x[n-10] if n-10 >= 0 else 0
        y[n] = xn + xn10
    return y

def sistema2(x):
    N = len(x)
    y = np.zeros(N)
    for n in range(N):
        xn = x[n]
        yn10 = 3*y[n-10] if n-10 >= 0 else 0
        y[n] = xn + yn10
    return y

#uso x1 como señal de entrada senoidal

# Respuesta al impulso de cada sistema
N = 50
delta = np.zeros(N)
delta[0] = 1
h1 = sistema1(delta)
h2 = sistema2(delta)

# Salidas a una senoidal
y1 = sistema1(x1)
y2 = sistema2(x1)

# Graficar
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
n = np.arange(len(h1))
plt.stem(n, h1)
plt.title("Respuesta al impulso h[n] - Sistema 1")

plt.subplot(2,2,2)
plt.plot(t[:len(y1)], y1)
plt.title("Salida para senoidal - Sistema 1")

plt.subplot(2,2,3)
n = np.arange(len(h2))
plt.stem(n, h2)
plt.title("Respuesta al impulso h[n] - Sistema 2")

plt.subplot(2,2,4)
plt.plot(t[:len(y2)], y2)
plt.title("Salida para senoidal - Sistema 2")

plt.tight_layout()
plt.show()
