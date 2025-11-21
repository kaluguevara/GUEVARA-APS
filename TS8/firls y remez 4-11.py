# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 23:24:41 2025

@author: Carola
"""

#CODIGO COMPLETO con firls y remez

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
#from matplotlib import patches
from pytc2.sistemas_lineales import plot_plantilla
#import scipy.io as sio

# Plantilla de diseño 
# pasabanda digital 

fs = 1000 #[Hz]

alpha_p = 1 #atenuacion maxima a la wp
alpha_s = 40 #atenuacion minima a la ws

#%% #################
#Diseño de filtro FIR
#####################

wp = (0.8, 35) #comienzo y fin banda de paso
ws = (0.1, 35.7)  #banda de stop [Hz]
#debería armar un wp1 y un wp2, para que no estén ambas juntas cuando quiero hacer una predistorsión 

frecuencias = np.sort(np.concatenate(((0, fs/2), wp, ws)))
deseado = [0, 0, 1, 1, 0, 0]

cant_coef = 3701 #debe ser impar y mayorigual a 1, lo aumenté de 2001 a 3001 y cumplió la plantilla
retardo = (cant_coef - 1) // 2

#FIR CON CUADRADOS MÍNIMOS
#fir_win_rectangular = sig.firwin2(numtaps=cant_coef,freq=frecuencias, gain=deseado, window='boxcar', fs=fs, nfreqs=int((np.ceil(np.sqrt(cant_coef)*2)**2)-1))
fir_ls = sig.firls(numtaps=cant_coef,bands=frecuencias, desired=deseado, fs=fs)

#FIR CON PARKS-MCCLELLAN (REMEZ)
bands_remez = [0, ws[0], wp[0], wp[1], ws[1], fs/2]
desired_remez = [0, 1, 0]
weight_remez = [1,1, 1]  #le da menos peso a la bada de paso
fir_remez=sig.remez(numtaps=cant_coef, bands=bands_remez, desired=desired_remez, weight=weight_remez, fs=fs)
w_ls, h_ls = sig.freqz(b=fir_ls, worN=np.logspace(-2, 2, 3000), fs=fs)
w_remez, h_remez = sig.freqz(b=fir_remez, worN=np.logspace(-2, 2, 3000), fs=fs)


for nombre, w, h in [("FIR Least Squares", w_ls, h_ls), ("FIR Remez", w_remez, h_remez)]:
    
    fase = np.unwrap(np.angle(h))
    w_rad = w / (fs / 2) * np.pi
    gd = -np.diff(fase) / np.diff(w_rad)
    
    plt.figure(figsize=(12,10))

    # Magnitud
    plt.subplot(3,1,1)
    plt.plot(w, 20*np.log10(abs(h)), label=nombre)
    plot_plantilla(filter_type='bandpass', fpass=wp, ripple=alpha_p, fstop=ws, attenuation=alpha_s, fs=fs)
    plt.title('Respuesta en Magnitud')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('|H(ω)| [dB]')
    plt.grid(True, which='both', ls=':')
    plt.legend()

    # Fase
    plt.subplot(3,1,2)
    plt.plot(w, fase, label=nombre)
    plt.title('Fase')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Fase [rad]')
    plt.grid(True, which='both', ls=':')
    plt.legend()

    # Retardo de grupo
    plt.subplot(3,1,3)
    plt.plot(w[1:], gd, label=nombre)
    plt.title('Retardo de Grupo')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('τg [# muestras]')
    plt.grid(True, which='both', ls=':')
    plt.legend()

#-------------------------------
# Comparación de ambos sobre la plantilla
#-------------------------------

plt.figure(figsize=(10,6))
plt.plot(w_ls, 20*np.log10(abs(h_ls)), label='FIR Least Squares')
plt.plot(w_remez, 20*np.log10(abs(h_remez)), label='FIR Remezzz', linestyle='--')
plot_plantilla(filter_type='bandpass', fpass=wp, ripple=alpha_p, fstop=ws, attenuation=alpha_s, fs=fs)
plt.title('Comparación sobre la plantilla')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(ω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.show()


# si sigo teniendo probemas para hacerlo converger, parto el diseño a la mitad. planteo dos diseños separados (por ejemplo un pasaalto más un pasabajos). 
# pero cómo los sumo? 
# convoluciono las dos respuestas al impulso, ojo con que me va a quedar un coeficiente más (la suma de las 2 +1) (??)
#lo de polos y ceros a tomar por culo me cansé de comentarlo
#lo que siempre quiero hacer es relajar la pendiente de transición

#%% ###################################
# Lectura y filtrado de señal ECG
###################################

# fs_ecg = 1000 #Hz

# mat_struct = sio.loadmat('./ECG_TP4.mat')
# ecg_one_lead = mat_struct['ecg_lead'].flatten()
# N = len(ecg_one_lead)
# cant_muestras = N

# # Filtrado FIR
# ecg_filt_win = sig.lfilter(b=fir_ls, a=1, x=ecg_one_lead)

# #%% Visualización general
# plt.figure()
# plt.plot(ecg_one_lead, label='ECG crudo')
# plt.plot(ecg_filt_win, label='FIR Window')
# plt.legend()

# ###################################
# # Regiones de interés sin ruido
# ###################################

# regs_interes = (
#     [4000, 5500],
#     [10e3, 11e3],
# )

# for ii in regs_interes:
#     zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')

#     plt.figure()
#     plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
#     plt.plot(zoom_region, ecg_filt_win[zoom_region + retardo], label='FIR Window')

#     plt.title('ECG sin ruido desde ' + str(ii[0]) + ' a ' + str(ii[1]))
#     plt.ylabel('Adimensional')
#     plt.xlabel('Muestras (#)')

#     axes_hdl = plt.gca()
#     axes_hdl.legend()
#     axes_hdl.set_yticks(())

#     plt.show()

# ###################################
# # Regiones de interés con ruido
# ###################################

# regs_interes = (
#     np.array([5, 5.2]) * 60 * fs,
#     np.array([12, 12.4]) * 60 * fs,
#     np.array([15, 15.2]) * 60 * fs,
# )

# for ii in regs_interes:
#     zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')

#     plt.figure()
#     plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
#     plt.plot(zoom_region, ecg_filt_win[zoom_region + retardo], label='FIR Window')

#     plt.title('ECG con ruido desde ' + str(ii[0]) + ' a ' + str(ii[1]))
#     plt.ylabel('Adimensional')
#     plt.xlabel('Muestras (#)')

#     axes_hdl = plt.gca()
#     axes_hdl.legend()
#     axes_hdl.set_yticks(())

#     plt.show()
