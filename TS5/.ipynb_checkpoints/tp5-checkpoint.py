# -*- coding: utf-8 -*- 
"""
Created on Mon Oct 13 15:57:33 2025

@author: gueva
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio

# FUNCION PARA ESTIMAR ANCHO DE BANDA PARA EL PUNTO 2
def estimar_ancho_de_banda(f, Pxx, porcentaje=0.995):
    # Normalizar la densidad espectral para obtener energía relativa
    energia_acum = np.cumsum(Pxx)
    energia_acum_norm = energia_acum / energia_acum[-1]
    # Buscar la frecuencia donde se alcanza el porcentaje deseado (por defecto 99.5%)
    indice_corte = np.where(energia_acum_norm >= porcentaje)[0][0]
    f_corte = f[indice_corte]
    # El ancho de banda va desde 0 Hz hasta esa frecuencia de corte
    ancho_banda = f_corte - f[0]
    return ancho_banda, f[0], f_corte


#%% ECG
fs = 1000#Hz
ecg = np.load('ecg_sin_ruido.npy')

f, Pxx = sig.welch(ecg, fs=fs, window='hann', nperseg=fs//2, noverlap=fs//4, scaling='density')
t = np.arange(len(ecg)) / fs

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, ecg, color='red')
plt.title("Señal ECG - Dominio Temporal")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogy(f, Pxx, color='red')
plt.title("Densidad Espectral de Potencia (Método de Welch)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [V²/Hz]")
plt.grid(True)
plt.tight_layout()
plt.show()

bw_ecg, fmin_ecg, fmax_ecg = estimar_ancho_de_banda(f, Pxx)
print("\nANCHO DE BANDA ESTIMADO:")
print("________________________________________________________________________________________")
print(f"[ECG]        | {bw_ecg:.2f} Hz (desde {fmin_ecg:.2f} Hz hasta {fmax_ecg:.2f} Hz)")

#%% PPG 
fs = 400#Hz
ppg = np.load('ppg_sin_ruido.npy')

f, Pxx = sig.welch(ppg, fs=fs, window='hann', nperseg=fs//2, noverlap=fs//4, scaling='density')
t = np.arange(len(ppg)) / fs

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, ppg, color='hotpink')
plt.title("Señal PPG - Dominio Temporal")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.xlim(0,70)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogy(f, Pxx, color='hotpink')
plt.title("Densidad Espectral de Potencia (Método de Welch)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [V²/Hz]")
plt.grid(True)
plt.tight_layout()
plt.show()

bw_ppg, fmin_ppg, fmax_ppg = estimar_ancho_de_banda(f, Pxx)
print(f"[PPG]        | {bw_ppg:.2f} Hz (desde {fmin_ppg:.2f} Hz hasta {fmax_ppg:.2f} Hz)")

#%% AUDIOS DEL SILBIDO, CUCA Y MARIANO HABLANDO
fs1, wav1 = sio.wavfile.read('la cucaracha.wav')
fs2, wav2 = sio.wavfile.read('prueba psd.wav')
fs3, wav3 = sio.wavfile.read('silbido.wav')

#Convertir a mono si es estereo, de dos a 1 canal, asi no se me duplica la energia
if wav1.ndim > 1: wav1 = wav1[:, 0]
if wav2.ndim > 1: wav2 = wav2[:, 0]
if wav3.ndim > 1: wav3 = wav3[:, 0]

def analizar_audio(senal, fs, nombre, color):
    f, Pxx = sig.welch(senal, fs=fs, window='hann', nperseg=fs//2, noverlap=fs//4, scaling='density')
    t = np.arange(len(senal)) / fs

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, senal, color=color)
    plt.title(f"Señal de audio - {nombre}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.semilogy(f, Pxx, color=color)
    plt.title("Densidad Espectral de Potencia (Método de Welch)")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("PSD [V²/Hz]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    bw, fmin, fmax = estimar_ancho_de_banda(f, Pxx)
    print(f"[{nombre}]   | {bw:.2f} Hz (desde {fmin:.2f} Hz hasta {fmax:.2f} Hz)")

analizar_audio(wav1, fs1, "La Cucaracha", "darkorange")
analizar_audio(wav2, fs2, "Prueba PSD", "limegreen")
analizar_audio(wav3, fs3, "Silbido", "royalblue")
