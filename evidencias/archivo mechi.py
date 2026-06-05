#%% Librerías
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal as sig
import scipy.io as sio

#%% Ejemplo 1: Módulo en veces y Fase desenvuelta
fs = 1000      # Frecuencia de muestreo del filtro en Hz
gpass = 1     # Atenuación máxima permitida en banda de paso (dB)
gstop = 40    # Atenuación mínima requerida en banda de stop (dB)

# Para un filtro pasa banda (diseño de plantilla - actualizados a captura 212335)
ws1 = .1
wp1 = .5 
ws2 = 45
wp2 = 35

wp = [wp1, wp2]
ws = [ws1, ws2] 

ftype = 'butter' 

# Diseño del filtro para filtrado bidireccional (filtfilt) con tolerancias a la mitad
sos_but = sig.iirdesign(wp, ws, gpass/2, gstop/2, fs=fs, analog=False, ftype='butter', output='sos')
sos_cauer = sig.iirdesign(wp, ws, gpass/2, gstop/2, fs=fs, analog=False, ftype='cauer', output='sos')
sos_cheby2 = sig.iirdesign(wp, ws, gpass/2, gstop/2, fs=fs, analog=False, ftype='cheby2', output='sos')

# Vector de frecuencias personalizado
ww = np.concatenate([np.logspace(start=-2, stop=0.1, num=500),
                np.linspace(start=1.26, stop=35, num=200),
                np.logspace(start=1.55, stop=1.65, num=300),
                np.linspace(start=46, stop=fs//2, num=50)])

# Respuesta en frecuencia evaluada en el vector ww
w, h = sig.sosfreqz(sos_but, worN=ww, fs=fs)

fig, ax1 = plt.subplots(tight_layout=True)
ax1.set_title("Respuesta en Frecuencia del Filtro IIR (Butterworth)")

# Gráfico del Módulo (En veces de forma lineal)
ax1.plot(w, abs(h), 'C0', label='Módulo')
ax1.set_ylabel("Módulo [Veces]", color='C0')
ax1.set_xlabel("Frecuencia [Hz]")
ax1.tick_params(axis='y', labelcolor='C0')
ax1.grid(True)

# Gráfico de la Fase (En radianes)
ax2 = ax1.twinx()
phase = np.unwrap(np.angle(h)) 
ax2.plot(w, phase, 'C1', label='Fase')
ax2.set_ylabel('Fase [rad]', color='C1')
ax2.tick_params(axis='y', labelcolor='C1')

plt.show()

#%% Clase 6/4/2026: Diseño FIR
numtaps = 3000
freq = np.array([0., ws1, wp1, wp2, ws2, fs/2])
gains = np.array([0, 0, 1, 1, 0, 0])

b_win = sig.firwin2(numtaps=numtaps, freq=freq, gain=gains, window="boxcar", fs=fs)
w_fir, h_fir = sig.freqz(b_win, worN=ww, fs=fs)
ceros_fir, polos_fir, k_fir = sig.tf2zpk(b_win, a=1)

#%% Ejemplo 2: Respuesta con Módulo en Decibeles [dB] e índices alineados Y PLANTILLA
w_but, h_but = sig.sosfreqz(sos_but, worN=ww, fs=fs)

plt.figure(figsize=(10, 5), tight_layout=True)
plt.title('Comparativa IIR vs FIR y Plantilla de Diseño')
plt.plot(w_but, 20*np.log10(np.maximum(np.abs(h_but), 1e-10)), label='Butterworth')
plt.plot(w_fir, 20*np.log10(np.maximum(np.abs(h_fir), 1e-10)), label='FIR Boxcar')

# Generación de la plantilla con un estilo propio (sombreado gris diagonal)
plt.fill_between([0, ws1], -gstop, 5, color='gray', alpha=0.2, hatch='//', label='Banda de Rechazo')
plt.fill_between([ws2, fs/2], -gstop, 5, color='gray', alpha=0.2, hatch='//')
plt.fill_between([wp1, wp2], -120, -gpass, color='gray', alpha=0.2, hatch='\\', label='Banda de Paso')

plt.axhline(-gpass, color='black', linestyle=':', linewidth=1.5)
plt.axhline(-gstop, color='black', linestyle=':', linewidth=1.5)

plt.legend(loc='lower center', ncol=4, fontsize=9)
plt.grid(True, linestyle='--')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.ylim(-100, 5)
plt.xlim(0, fs/2)
plt.show()

# =============================================================================
# DIAGRAMA DE POLOS Y CEROS Y RETARDO DE GRUPO
# =============================================================================
fig, ax = plt.subplots(figsize=(6,6))

circulo = plt.Circle((0,0), 1, color='gray', fill=False, linestyle='--')
ax.add_artist(circulo)

ax.plot(np.real(ceros_fir), np.imag(ceros_fir), 'bo', markersize=8, fillstyle='none', label='Ceros')
ax.plot(np.real(polos_fir), np.imag(polos_fir), 'rx', markersize=8, mew=2, label='Polos')

ax.set_title('Diagrama de Polos y Ceros FIR')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-1.2,1.2)
ax.set_ylim(-1.2,1.2)
ax.grid(True)
ax.legend()
plt.show()

fig_sys = plt.figure(figsize=(12, 5), tight_layout=True)

# Panel Izquierdo: Plano Z IIR
ax_z = fig_sys.add_subplot(1, 2, 1)
ceros, polos, k = sig.sos2zpk(sos_but)
circulo = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1.5)
ax_z.add_artist(circulo)
ax_z.plot(np.real(ceros), np.imag(ceros), 'bo', markersize=8, fillstyle='none', label='Ceros')
ax_z.plot(np.real(polos), np.imag(polos), 'rx', markersize=8, mew=2, label='Polos')
ax_z.axhline(0, color='black', linewidth=0.5)
ax_z.axvline(0, color='black', linewidth=0.5)
ax_z.set_title('Diagrama de Polos y Ceros IIR (Plano z)')
ax_z.set_xlabel('Parte Real')
ax_z.set_ylabel('Parte Imaginaria')
ax_z.set_aspect('equal', adjustable='box')
ax_z.set_xlim([-1.2, 1.2])
ax_z.set_ylim([-1.2, 1.2])
ax_z.grid(True, alpha=0.5)
ax_z.legend()

# Panel Derecho: Retardo de Grupo Comparado
ax_gd = fig_sys.add_subplot(1, 2, 2)
# FIR
phase_fir = np.unwrap(np.angle(h_fir))
gd_fir = -np.diff(phase_fir)/np.diff(w_fir)
gd_fir = np.insert(gd_fir, 0, gd_fir[0])
# IIR
gd_iir = -np.diff(phase) / np.diff(ww)
gd_iir = np.append(gd_iir[0], gd_iir)

ax_gd.plot(ww, gd_iir, 'm', linewidth=2, label='IIR Butterworth')
ax_gd.plot(w_fir, gd_fir, 'C0--', label='FIR Boxcar')
ax_gd.set_title('Retardo de Grupo (Group Delay)')
ax_gd.set_xlabel('Frecuencia [Hz]')
ax_gd.set_ylabel('Retardo [Muestras]')
ax_gd.set_xlim(0, fs/2)
ax_gd.grid(True, alpha=0.5)
ax_gd.legend()
plt.show()

#%% Procesar el ECG
fs_ecg = 1000 # Hz

# Cargar variables del archivo
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
cant_muestras = len(ecg_one_lead)

# Filtrar el ECG con los filtros IIR
ECG_f_butt = sig.sosfiltfilt(sos_but, ecg_one_lead)
ECG_f_cauer = sig.sosfiltfilt(sos_cauer, ecg_one_lead)
ECG_f_cheby2 = sig.sosfiltfilt(sos_cheby2, ecg_one_lead)

# Filtrar el ECG con el filtro FIR y compensar demora temporal
ECG_f_win = sig.lfilter(b_win, 1, ecg_one_lead)
demora = int((numtaps - 1) / 2) 

###################################
#%% Regiones de interés con ruido #
###################################

regs_interes_ruido = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )

for ii in regs_interes_ruido:
    zoom_region = np.arange(np.max([0, int(ii[0])]), np.min([cant_muestras, int(ii[1])]), dtype='uint')
    
    plt.figure(figsize=(10, 4))
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG Original', linewidth=2, alpha=0.7)
    plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
    #plt.plot(zoom_region, ECG_f_cauer[zoom_region], label='Cauer')
    #plt.plot(zoom_region, ECG_f_cheby2[zoom_region], label='Cheby2')
    
    # Aplicando la compensación de la fase lineal al FIR para visualizarlo alineado
    plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window', linestyle='--')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Amplitud')
    plt.xlabel('Muestras (#)')
    plt.legend(loc='upper right', fontsize=8)
    plt.yticks([])
    plt.show()

###################################
#%% Regiones de interés sin ruido #
###################################

regs_interes_limpio = (
        np.array([5, 5.2]) * 60 * fs_ecg, 
        np.array([12, 12.4]) * 60 * fs_ecg, 
        np.array([15, 15.2]) * 60 * fs_ecg, 
        )

for ii in regs_interes_limpio:
    zoom_region = np.arange(np.max([0, int(ii[0])]), np.min([cant_muestras, int(ii[1])]), dtype='uint')
    
    plt.figure(figsize=(10, 4))
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG Original', linewidth=2, alpha=0.7)
    plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
    #plt.plot(zoom_region, ECG_f_cauer[zoom_region], label='Cauer')
    #plt.plot(zoom_region, ECG_f_cheby2[zoom_region], label='Cheby2')
    plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window', linestyle='--')
    
    plt.title('ECG filtering example from ' + str(int(ii[0])) + ' to ' + str(int(ii[1])) )
    plt.ylabel('Amplitud')
    plt.xlabel('Muestras (#)')
    plt.legend(loc='upper right', fontsize=8)
    plt.yticks([])
    plt.show()

