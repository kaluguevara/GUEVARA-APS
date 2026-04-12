# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:10:27 2026

@author: gueva
"""
import numpy as np
import matplotlib.pyplot as plt

#%%
#definimos valores
B = 4 #bits
Vcc = 2 #volts
kn = 1 #para escalar Pq
fs = 1000
N = 1000
frec = 50

#%%
#funciones
def funcion_seno(amp, desplazo, frec, fase, cant_muestras, frec_muestreo):
    tt = np.arange(cant_muestras) / frec_muestreo
    xx = amp * np.sin(2 * np.pi * frec * tt + fase) + desplazo
    return tt.reshape(-1,1), xx.reshape(-1,1)

#cuantizacion
def simular_adc(s, B, Vcc, kn, N, usar_ruido=True):
    # cuantización
    q = Vcc / (2**(B-1))
    Pq = (q**2) / 12

    # potencia de ruido
    Pn = kn * Pq
    mu = 0

    # ruido gaussiano
    n = np.random.normal(mu, np.sqrt(Pn), N).reshape(-1,1)

    # señal contaminada
    sr = s + n

    # selector de entrada al ADC
    if usar_ruido:
        s_in = sr
    else:
        s_in = s

    # cuantización
    sq = np.round(s_in / q) * q

    # error
    ee = sq - s_in

    return sr, sq, ee, q, Pq, Pn

#%%
#llamamos a las funciones
tt, s = funcion_seno(amp=1.8, desplazo=0, frec=fs/N, fase=0, cant_muestras=N, frec_muestreo=fs)
#uso esta frec para que me muestre un periodo
sr, sq, ee, q, Pq, Pn = simular_adc(s, B, Vcc, kn, N)
#%%
#PARTE 1
#le pedi al chat que me lo ponga al grafico fancy
plt.figure()
plt.plot(tt, sr, label=r'$s_R = s + n$ (ADC in)', linestyle='--', marker='.', markersize=3)
plt.plot(tt, sq, label=r'$s_Q$ (cuantizada)', linestyle='-', marker='o', markersize=3)
plt.plot(tt, s, label=r'$s$ (analog)', linewidth=2)
plt.title(f'{B} bits - ±Vr = {Vcc} V - q = {q:.3f} V')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
#lo rellamo cambiando la frecuencia
#%%
#PARTE 2

#por si se necesita una menor frecuencia
#tt, s = funcion_seno(amp=1.8, desplazo=0, frec=50, fase=0, cant_muestras=N, frec_muestreo=fs)
#sr, sq, ee, q, Pq, Pn = simular_adc(s, B, Vcc, kn, N)
#sq_hist = np.round(s / q) * q
#ee_hist = sq_hist - s

sq_hist = np.round(sr / q) * q
ee_hist = sq_hist - sr

#graficos
fig2, ax4 = plt.subplots(figsize=(6, 4))

#histograma
n_bins = 10
ax4.hist(ee_hist, bins=n_bins, density=True, alpha=0.7)
ax4.axvline(q/2, color='red', linestyle='--', linewidth=1)
ax4.axvline(-q/2, color='red', linestyle='--', linewidth=1)
#rectangulo lindo
ax4.hlines(1/q, -q/2, q/2, colors='red', linestyles='--', linewidth=1)
ax4.hlines(0, -q/2, q/2, colors='red', linestyles='--', linewidth=1)
#densidad teorica
ax4.axhline(1/q, color='orange', linestyle=':', label='Distribuicion uniforme teorica')
# labels
ax4.set_xlabel('Error de cuantización [V]')
ax4.set_ylabel('Densidad de probabilidad')
ax4.set_title(f'Ruido de cuantización {B} bits — ±Vref = {Vcc} V — q = {q:.4f} V')
ax4.set_xlim(-q/2, q/2)
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%
#PARTE 3
#fft
S  = np.fft.fft(s.flatten())
SR = np.fft.fft(sr.flatten())
SQ = np.fft.fft(sq.flatten())

#frec
f = np.fft.fftfreq(N, 1/fs)

# solo frecuencias positivas
idx = f >= 0

#densidad de potencia
PSD_s  = (np.abs(S)**2)  / (N * fs)
PSD_sr = (np.abs(SR)**2) / (N * fs)
PSD_sq = (np.abs(SQ)**2) / (N * fs)

#lo quiero en db
PSD_s_db  = 10 * np.log10(PSD_s  + 1e-12)
PSD_sr_db = 10 * np.log10(PSD_sr + 1e-12)
PSD_sq_db = 10 * np.log10(PSD_sq + 1e-12)

#le pido a chat que me grafique lindo
plt.figure(figsize=(8,4))

plt.plot(f[idx], PSD_s_db[idx], label='s (analógica)')
plt.plot(f[idx], PSD_sr_db[idx], '--', label='s + n')
plt.plot(f[idx], PSD_sq_db[idx], label='s_Q (cuantizada)')

# pisos
piso_analogico = 10*np.log10(Pn)
piso_digital = 10*np.log10(Pq)
plt.axhline(piso_analogico, linestyle='--', color='orange', label='Piso analógico')
plt.axhline(piso_digital, linestyle='--', color='red', label='Piso cuantización')

plt.title(f'Señal muestreada por ADC de {B} bits - ±Vr = {Vcc} V - q = {q:.3f} V')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de potencia [dB]')
plt.xlim(0, fs/2)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

