#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 20:46:52 2025
@author: victoria24
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
import scipy.io as sio

# Plantilla de diseño 
# pasabanda digital 

fs = 1000 #[Hz]
wp = [1, 35] #comienzo y fin banda de paso
ws = [0.01, 40]  #banda de stop [Hz]

#transicion en pocos hz implica orden mas elevado del polinomio 
 
alpha_p = 1 #atenuacion maxima a la wp, alpha max o pérdidas en banda de paso (dB)
alpha_s = 40 #atenuacion minima a la ws, alpha min o minima atenuacion requerida en banda de paso (dB)

#Aprox modulo 

f_aprox = 'cauer'
mi_sos_cauer = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs=fs)
#devuelve coeficientes del polinomio 

f_aprox = 'butter'
mi_sos_butter = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs=fs)

f_aprox = 'cheby1'
mi_sos_cheby1 = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs=fs)

f_aprox = 'cheby2'
mi_sos_cheby2 = signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs=fs)

# %%

# #w, h = signal.freqz_sos(mi_sos, worN=np.logspace(-2, 1.9, 1000), fs=fs)
# #espacio logaritmicamente espaciado. entre 10⁻2, 10⁶ y va a tomar 1000 puntos entre ambos

# w, h = signal.freqz_sos(mi_sos, fs=fs) # calcula la respuesta en frecuencia del filtro
# #w frecuencias donde evalua y h respuesta

# phase = np.unwrap(np.angle(h)) #unwrap es para que las discontinuidades evitables sean evitadas

# #Retardo de grupo = -delta phi / delta w
# w_rad = w / (fs/2) * np.pi 
# gd = -np.diff(phase) / np.diff(w_rad)

# #Polos y ceros
# z, p, k = signal.sos2zpk(mi_sos)   

# #graficos

# # --- Gráficas ---
# plt.figure()

# # Magnitud
# plt.subplot(2,2,1)
# plt.plot(w, 20*np.log10(abs(h)), label = f_aprox)
# plt.title('Respuesta en Magnitud')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('|H(jω)| [dB]')
# plt.grid(True, which='both', ls=':')
# plt.legend()

# # Fase
# plt.subplot(2,2,2)
# plt.plot(w, phase, label = f_aprox)
# plt.title('Fase')
# plt.xlabel('Pulsación angular [r/s]')
# plt.ylabel('Fase [°]')
# plt.grid(True, which='both', ls=':')
# plt.legend()

# # Retardo de grupo
# plt.subplot(2,2,3)
# plt.plot(w[:-1], gd, label = f_aprox)
# plt.title('Retardo de Grupo')
# plt.xlabel('Pulsación angular [r/s]')
# plt.ylabel('τg [(#muestras)]')
# plt.grid(True, which='both', ls=':')
# plt.legend()

# # Diagrama de polos y ceros
# plt.subplot(2,2,4)
# plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos')
# if len(z) > 0:
#     plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label='Ceros')
# plt.axhline(0, color='k', lw=0.5)
# plt.axvline(0, color='k', lw=0.5)
# plt.title('Diagrama de Polos y Ceros (plano s)')
# plt.xlabel('σ [rad/s]')
# plt.ylabel('jω [rad/s]')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

#%% 23-10-25
##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

#para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead'].squeeze()
N = len(ecg_one_lead)

#FILTRADO DEL ECG
#el caso de un ECG, donde importa mucho la posición temporal de los picos R, se suele preferir sosfiltfilt, porque mantiene la forma y el alineamiento de las ondas.

ecg_filt_butter = signal.sosfiltfilt(mi_sos_butter, ecg_one_lead)
ecg_filt_cauer = signal.sosfiltfilt(mi_sos_cauer, ecg_one_lead)
ecg_filt_cheby1 = signal.sosfiltfilt(mi_sos_cheby1, ecg_one_lead)
ecg_filt_cheby2 = signal.sosfiltfilt(mi_sos_cheby2, ecg_one_lead)

#con filtfilt estamos seguros de que no hay NIGUNA distorsión de fase
#aparece la respuesta al impulso en ambas direcciones

plt.figure()

plt.plot(ecg_one_lead[:50000], label='Original (ruidosa)', color='k', alpha=0.5, linewidth=1)
#plt.plot(ecg_filt_butter[:50000], label = 'butter')
#plt.plot(ecg_filt_cauer[:50000], label = 'cauer')
plt.plot(ecg_filt_cheby1[:50000], label = 'cheby1')
#plt.plot(ecg_filt_cheby2[:50000], label = 'cheby2')

plt.legend()
plt.show()

#%%
# Frecuencia de muestreo
fs = 1000  

# Calculamos respuesta en frecuencia para cada filtro
w, h_butter = signal.sosfreqz(mi_sos_butter, worN=np.logspace(-2, 1.9, 1000), fs=fs) #agrego puntos de muestreo donde me interesa
_, h_cauer  = signal.sosfreqz(mi_sos_cauer, worN=np.logspace(-2, 1.9, 1000), fs=fs)
_, h_cheby1 = signal.sosfreqz(mi_sos_cheby1, worN=np.logspace(-2, 1.9, 1000), fs=fs)
_, h_cheby2 = signal.sosfreqz(mi_sos_cheby2, worN=np.logspace(-2, 1.9, 1000), fs=fs)

plt.figure(figsize=(10,6))
plt.plot(w, 20*np.log10(np.maximum(abs(h_butter), 1e-8)), label='Butterworth')
plt.plot(w, 20*np.log10(np.maximum(abs(h_cauer), 1e-8)), label='Cauer')
plt.plot(w, 20*np.log10(np.maximum(abs(h_cheby1), 1e-8)), label='Chebyshev I')
plt.plot(w, 20*np.log10(np.maximum(abs(h_cheby2), 1e-8)), label='Chebyshev II')

# plt.axvspan(1, 35, color='green', alpha=0.1, label='Banda de paso')
# plt.axvspan(40, fs/2, color='red', alpha=0.1, label='Banda de rechazo alta')

plt.ylim(-100, 5) #evitar valores demasiado grandes en dB
plt.title('Respuesta en frecuencia de los filtros IIR')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout()
plt.show()

#%% polos y ceros

# Convertir de SOS a ZPK
z_butter, p_butter, k_butter = signal.sos2zpk(mi_sos_butter)
z_cauer, p_cauer, k_cauer = signal.sos2zpk(mi_sos_cauer)
z_cheby1, p_cheby1, k_cheby1 = signal.sos2zpk(mi_sos_cheby1)
z_cheby2, p_cheby2, k_cheby2 = signal.sos2zpk(mi_sos_cheby2)

# Preparar figura
plt.figure(figsize=(10,10))

# Círculo unitario
theta = np.linspace(0, 2*np.pi, 200)
plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)

# Graficar polos y ceros de cada filtro
plt.plot(np.real(p_butter), np.imag(p_butter), 'x', markersize=10, label='Polos Butterworth')
plt.plot(np.real(z_butter), np.imag(z_butter), 'o', markersize=10, fillstyle='none', label='Ceros Butterworth')

plt.plot(np.real(p_cauer), np.imag(p_cauer), 'x', markersize=10, label='Polos Cauer')
plt.plot(np.real(z_cauer), np.imag(z_cauer), 'o', markersize=10, fillstyle='none', label='Ceros Cauer')

plt.plot(np.real(p_cheby1), np.imag(p_cheby1), 'x', markersize=10, label='Polos Cheby I')
plt.plot(np.real(z_cheby1), np.imag(z_cheby1), 'o', markersize=10, fillstyle='none', label='Ceros Cheby I')

plt.plot(np.real(p_cheby2), np.imag(p_cheby2), 'x', markersize=10, label='Polos Cheby II')
plt.plot(np.real(z_cheby2), np.imag(z_cheby2), 'o', markersize=10, fillstyle='none', label='Ceros Cheby II')

plt.xlabel('Re')
plt.ylabel('Im')
plt.title('Diagrama de Polos y Ceros de los Filtros IIR')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

#################################
# Regiones de interés sin ruido #
#################################

cant_muestras = len(ecg_one_lead)

regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_filt_butter[zoom_region], label='Butterworth')
    #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG sin ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
 
#################################
# Regiones de interés con ruido #
#################################
 
regs_interes = (
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_filt_butter[zoom_region], label='Butterworth')
    # plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG con ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()