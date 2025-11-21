# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 12:48:32 2025

@author: gueva
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
from pytc2.sistemas_lineales import plot_plantilla
import warnings
warnings.filterwarnings("ignore", message="Badly conditioned filter coefficients")
#%% CAMBIOS ESTETICOS
plt.style.use('default')  #se resetea
#defino colores
COLOR_FONDO = '#FFF0F5'  
COLOR_VERDE = '#2E8B57'   
COLOR_ROSA = '#FF69B4'    

# %% VARIABLES IIR PARA LA PLANTILLA

fs = 1000 # Hz
fs_ecg = 1000 # Hz
wp = (0.8, 35) #frecuencia de paso
ws = (0.1, 40) #frecuencia de rechazo 
alpha_p = 1/2 #atenuacion de banda de paso / el alfa maximo
alpha_s = 40/2 #atenuacion de banda de rechazo / el alfa minimo

#abro archivo
mat_struct = sio.loadmat('./ecg.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)

#%%
def filtro_IIR(fs, wp, ws, alpha_p, alpha_s, ftype): 
    
    mi_sos = signal.iirdesign(wp = wp, ws = ws, gpass = alpha_p, gstop = alpha_s, analog = False, ftype = ftype, output ='sos', fs=fs)
    
    #calculo rta en frec
    w, h= signal.freqz_sos(mi_sos, worN = np.logspace(-2, 1.9, 1000), fs = fs)
    fase = np.unwrap(np.angle(h)) 
    w_rad = w / (fs / 2) * np.pi
    gd = -np.diff(fase) / np.diff(w_rad) 
    
    #politos y ceritos con la funcion que vimos
    z, p, k = signal.sos2zpk(mi_sos)
    
    #magnitud
    plt.figure(figsize=(8, 8))
    plt.subplot(3,1,1)
    plt.plot(w, 20 * np.log10(np.maximum(abs(h), 1e-10)), 
             label=ftype, color=COLOR_ROSA, linewidth=2)
    plot_plantilla(filter_type='bandpass', fpass=wp, ripple=alpha_p*2, 
                   fstop=ws, attenuation=alpha_s*2, fs=fs)
    plt.title('Respuesta en Magnitud', fontweight='bold')
    plt.ylabel('|H(jω)| [dB]')
    plt.xlabel('Frecuencia [Hz]')
    plt.grid(True, which='both', ls=':', alpha=0.7)
    plt.legend()

    #fase
    plt.subplot(3,1,2)
    plt.plot(w, np.degrees(fase), color=COLOR_ROSA, linewidth=2)
    plt.title('Fase', fontweight='bold')
    plt.ylabel('Fase [°]')
    plt.xlabel('Frecuencia [Hz]')
    plt.grid(True, which='both', ls=':', alpha=0.7)
    
    #retardo de grupo
    plt.subplot(3,1,3)
    plt.plot(w[:-1], gd, color=COLOR_ROSA, linewidth=2)
    plt.title('Retardo de Grupo', fontweight='bold')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('τg [# muestras]')
    plt.grid(True, which='both', ls=':', alpha=0.7)
    
    #pone fondito pink
    plt.gcf().patch.set_facecolor(COLOR_FONDO)
    plt.tight_layout()
    plt.show()
    
    return mi_sos

def filtrar_IIR_ECG(mi_sos, nombre_filtro, ecg=ecg_one_lead, fs=fs_ecg):
    ecg_filt = signal.sosfiltfilt(mi_sos, ecg)

    plt.figure()
    plt.plot(ecg, label='ecg crudo', alpha=0.7, color='forestgreen')
    plt.plot(ecg_filt, label=f'Filtrado ({nombre_filtro})', 
             linewidth=1.5, color=COLOR_ROSA)
    plt.title(f'ECG completo - {nombre_filtro}', fontweight='bold')
    plt.xlabel('Muestras (#)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True, alpha=0.7)
    plt.gcf().patch.set_facecolor(COLOR_FONDO)
    plt.show()
    
#COPY PASTE DE LA PARTE DE MARIANO DEL CAMPUS
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
        plt.plot(zoom_region, ecg[zoom_region], label='ECG', linewidth=2)
        plt.plot(zoom_region, ecg_filt[zoom_region], label=nombre_filtro)
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
        plt.plot(zoom_region, ecg[zoom_region], label='ECG', linewidth=2)
        plt.plot(zoom_region, ecg_filt[zoom_region], label=nombre_filtro)
        # plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
       
        plt.title('ECG con ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
        plt.ylabel('Adimensional')
        plt.xlabel('Muestras (#)')
       
        axes_hdl = plt.gca()
        axes_hdl.legend()
        axes_hdl.set_yticks(())
               
        plt.show()

#para comparar ambos filtros fir
tipos_iir = ['butter', 'ellip']  #ellip es la cauer
resultados_iir = {}

for tipo_actual in tipos_iir:
    coeficientes_sos = filtro_IIR(
        fs=fs,
        wp=wp, 
        ws=ws,
        alpha_p=alpha_p,
        alpha_s=alpha_s,
        ftype=tipo_actual
    )
    resultados_iir[tipo_actual] = coeficientes_sos

#filtro cada metodo
for nombre_tipo, coef_sos in resultados_iir.items():
    filtrar_IIR_ECG(coef_sos, nombre_tipo)