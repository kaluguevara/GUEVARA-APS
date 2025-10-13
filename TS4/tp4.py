# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 21:12:20 2025

@author: gueva
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import scipy.signal.windows as win

#DEFINICION DE FUNCIONES
def senial(tt, frec, amp, fase=0, vmed=0):
    return amp * np.sin(2 * np.pi * frec * tt + fase) + vmed

def senial_mas_ruido(s, SNR_dB):        
    senial_p = np.mean(s**2)   
    ruido_p = senial_p / (10**(SNR_dB/10))
    ruido = np.random.normal(0, np.sqrt(ruido_p), size=s.shape)
    return s + ruido

#PARÁMETROS DE SIMULACIÓN
N = 1000           #cant de muestras
fs = 1000          #frec de muestreo en Hz
R = 200            #cant de repeticiones
df = fs / N        #resolución espectral
frecuencias = np.arange(N) * df
t = np.arange(N) / fs

#MATRICES 
t_mat = np.tile(t.reshape(N, 1), (1, R))
f_variacion = np.random.uniform(-2, 2, R)
f_base = (N / 4 + f_variacion) * df
f_mat = np.tile(f_base.reshape(1, R), (N, 1))

#GENERO SEÑALES
A_real = np.sqrt(2)
x = senial(t_mat, f_mat, A_real)
x_SNR_BAJO = senial_mas_ruido(x,SNR_dB=3)
x_SNR_ALTO = senial_mas_ruido(x,SNR_dB=10)

#VENTANAS Y FFT
ventanas = {
    "Rectangular": np.ones((N, 1)),
    "Flattop": win.flattop(N, sym=False).reshape(-1, 1),
    "Blackman-Harris": win.blackmanharris(N, sym=False).reshape(-1, 1),
    "Hann": win.hann(N, sym=False).reshape(-1, 1)
}

def calcular_fft(x, ventana):
    return (1/N) * fft(x * ventana, axis=0)

fft_bajo = {}
fft_alto = {}

for nombre, vent in ventanas.items():
    fft_bajo[nombre] = calcular_fft(x_SNR_BAJO, vent)
    fft_alto[nombre] = calcular_fft(x_SNR_ALTO, vent)

#ESTIMACIÓN DE AMPLITUD
def estimar_amp(X, ventana, f_objetivo, fs):
    idx = (f_objetivo * N / fs).astype(int)
    cg = np.sum(ventana) / N                 
    return 2 * np.abs(X[idx, np.arange(X.shape[1])]) / cg

amplitudes_bajo = {}
amplitudes_alto = {}

for nombre in fft_bajo.keys():
    amplitudes_bajo[nombre] = estimar_amp(fft_bajo[nombre], ventanas[nombre], f_base, fs)
for nombre in fft_alto.keys():  
    amplitudes_alto[nombre] = estimar_amp(fft_alto[nombre], ventanas[nombre], f_base, fs)

#SESGO Y VARIANZA
def obtener_metricas(amplitudes, A_teorica):
    sesgo = np.mean(amplitudes) - A_teorica
    varianza = np.var(amplitudes)
    return sesgo, varianza

metricas_bajo = {}
metricas_alto = {}

for nombre in amplitudes_bajo.keys():
    metricas_bajo[nombre] = obtener_metricas(amplitudes_bajo[nombre], A_real)

for nombre in amplitudes_alto.keys():
    metricas_alto[nombre] = obtener_metricas(amplitudes_alto[nombre], A_real)
  
#ESTIMADOR DE FRECUENCIA
def estimar_frec(X, fs, f_base):
    N = X.shape[0]                              
    indices = np.argmax(np.abs(X[:N//2, :]), axis=0)
    frecs_est = indices * fs / N
    sesgo = np.mean(frecs_est - f_base)
    varianza = np.var(frecs_est)
    return sesgo, varianza

frec_bajo = {}
for k, v in fft_bajo.items():
    frec_bajo[k] = estimar_frec(v, fs, f_base)

frec_alto = {}
for k, v in fft_alto.items():
    frec_alto[k] = estimar_frec(v, fs, f_base)

#TABLAS DE RESULTADOS
print("\nResultados (SNR = 3 dB) - Estimación de AMPLITUD")
print(f"{'Ventana':<22} | {'Sesgo [V]':>12} | {'Varianza [V²]':>15}")
print("-" * 55)
for nombre, (sesgo, varianza) in metricas_bajo.items():
    print(f"{nombre:<22} | {sesgo:>12.6f} | {varianza:>15.8f}")

print("\nResultados (SNR = 10 dB) - Estimación de AMPLITUD")
print(f"{'Ventana':<22} | {'Sesgo [V]':>12} | {'Varianza [V²]':>15}")
print("-" * 55)
for nombre, (sesgo, varianza) in metricas_alto.items():
    print(f"{nombre:<22} | {sesgo:>12.6f} | {varianza:>15.8f}")

print("\nResultados (SNR = 3 dB) - Estimación de FRECUENCIA")
print(f"{'Ventana':<22} | {'Sesgo [Hz]':>12} | {'Varianza [Hz²]':>15}")
print("-" * 55)
for nombre, (sesgo, varianza) in frec_bajo.items():
    print(f"{nombre:<22} | {sesgo:>12.6f} | {varianza:>15.8f}")

print("\nResultados (SNR = 10 dB) - Estimación de FRECUENCIA")
print(f"{'Ventana':<22} | {'Sesgo [Hz]':>12} | {'Varianza [Hz²]':>15}")
print("-" * 55)
for nombre, (sesgo, varianza) in frec_alto.items():
    print(f"{nombre:<22} | {sesgo:>12.6f} | {varianza:>15.8f}")


#GRÁFICOS DE PSD
f_central = (N / 4) * df
margen = 50  

# PSD para SNR bajo
for nombre, X in fft_bajo.items():
    plt.figure(figsize=(6, 4))
    plt.plot(frecuencias, 10 * np.log10(np.abs(X)**2), lw=0.7)
    plt.title(f"PSD (SNR = 3 dB) - Ventana {nombre}")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Potencia [dB]")
    plt.xlim(f_central - margen, f_central + margen)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# PSD para SNR alto
for nombre, X in fft_alto.items():
    plt.figure(figsize=(6, 4))
    plt.plot(frecuencias, 10 * np.log10(np.abs(X)**2), lw=0.7)
    plt.title(f"PSD (SNR = 10 dB) - Ventana {nombre}")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Potencia [dB]")
    plt.xlim(f_central - margen, f_central + margen)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
