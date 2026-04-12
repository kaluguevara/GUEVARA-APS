# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:10:27 2026

@author: gueva
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:10:27 2026

@author: gueva
"""
#%% =========================
# PARTE (b) — señal en fs/4
# =========================

import numpy as np
import matplotlib.pyplot as plt

# parámetros
B = 4
Vcc = 2
fs = 1000
N = 1000
kn = 1

# paso de cuantización
q = (2 * Vcc) / (2**B)
Pq = (q**2) / 12

# ruido analógico
Pn = kn * Pq

# señal en fs/4
f0 = fs / 4

def funcion_seno(amp, desplazo, frec, fase, cant_muestras, frec_muestreo):
    tt = np.arange(cant_muestras) / frec_muestreo
    xx = amp * np.sin(2 * np.pi * frec * tt + fase) + desplazo
    return tt.reshape(-1,1), xx.reshape(-1,1)

tt, s = funcion_seno(
    amp=1,
    desplazo=0,
    frec=f0,
    fase=0,
    cant_muestras=N,
    frec_muestreo=fs
)

# cuantización
sq = np.round(s / q) * q

# FFT
SQ = np.fft.fft(sq.flatten())
f = np.fft.fftfreq(N, 1/fs)

# solo frecuencias positivas
mask = f >= 0
f_pos = f[mask]

PSD_sq = (np.abs(SQ)**2)[mask] / (N * fs)
PSD_sq_db = 10 * np.log10(PSD_sq + 1e-12)

# pisos de ruido
piso_analogico = 10 * np.log10(Pn)
piso_digital = 10 * np.log10(Pq)

# gráfico
plt.figure(figsize=(8,4))

plt.plot(f_pos, PSD_sq_db, label='Señal cuantizada (fs/4)')

plt.axhline(piso_analogico, linestyle='--', color='orange', label='Piso ruido analógico')
plt.axhline(piso_digital, linestyle='--', color='red', label='Piso ruido cuantización')

plt.title(f'ADC {B} bits — Señal en fs/4 = {f0} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de potencia [dB]')
plt.xlim(0, fs/2)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()