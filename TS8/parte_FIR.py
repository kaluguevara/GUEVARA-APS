# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:48:36 2025

@author: gueva
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
from scipy.signal import firwin2, freqz, firls
from pytc2.sistemas_lineales import plot_plantilla

#%% CAMBIOS ESTETICOS
plt.style.use('default')  #se resetea
#defino colores
COLOR_FONDO = '#FFF0F5'  
COLOR_VERDE = '#2E8B57'   
COLOR_ROSA = '#FF69B4'    

# %% VARIABLES FIR PARA LA PLANTILLA

fs = 1000 # Hz
fs_ecg = 1000 # Hz
alpha_p = 1/2 #atenuacion de banda de paso / el alfa maximo
alpha_s = 40/2 #atenuacion de banda de rechazo / el alfa minimo
wp = (0.95, 35) #frecuencia de paso, va en rad/s
ws = (0.14, 35.7) #frecuencia de rechazo, va en rad/s

#abro archivo
mat_struct = sio.loadmat('./ecg.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)

#%%
def filtro_FIR(fs, wp, ws, alpha_p, alpha_s, metodo='firwin2'):
    frecuencias = [0, ws[0], wp[0], wp[1], ws[1], fs/2]

    deseado = [0,0,1,1,0,0]
    cant_coef = 2001 if metodo == 'firwin2' else 1999  #esto es por si es impar para fase lineal
    retardo = (cant_coef-1)//2 
    peso = [12,4,4]

    if metodo == 'firwin2':
        b = firwin2(numtaps=cant_coef, freq=frecuencias, gain=deseado, window='hamming', fs=fs)
    elif metodo == 'firls':
        b = firls(numtaps=cant_coef, bands=frecuencias, desired=deseado, fs=fs, weight= peso)
    else:
        raise ValueError("Método inválido.")
    
    #calculo rta en frec
    w, h= freqz(b, worN = np.logspace(-2, 2, 1000), fs = fs)   
    fase = np.unwrap(np.angle(h)) 
    w_rad = 2*np.pi*w/fs
    gd = -np.diff(fase) / np.diff(w_rad) 
    
    #politos y ceritos con la funcion que vimos
    z, p, k = signal.sos2zpk(signal.tf2sos(b,a= 1))
    
    #PLOTEOOO
    plt.figure(figsize=(8,8))
    
    #magnitud
    plt.subplot(3,1,1)
    plt.plot(w, 20*np.log10(np.maximum(np.abs(h), 1e-10)), 
             label=metodo, color=COLOR_ROSA, linewidth=2)
    plot_plantilla(filter_type='bandpass', fpass=(0.8, 35), ripple=alpha_p*2, 
                   fstop=(0.1, 40), attenuation=alpha_s*2, fs=fs)
    plt.title('Respuesta en Magnitud', fontweight='bold')
    plt.ylabel('|H(z)| [dB]')
    plt.xlabel('[Hz]')
    plt.grid(True, which='both', ls=':', alpha=0.7)
    plt.legend()

    #fase
    plt.subplot(3,1,2)
    plt.plot(w, np.degrees(fase), color=COLOR_ROSA, linewidth=2)
    plt.title('Fase', fontweight='bold')
    plt.ylabel('Fase [°]')
    plt.xlabel('[Hz]')
    plt.grid(True, which='both', ls=':', alpha=0.7)
    
    #retardo de grupo
    plt.subplot(3,1,3)
    plt.plot(w[:-1], gd, color=COLOR_ROSA, linewidth=2)
    plt.title('Retardo de Grupo', fontweight='bold')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('τg [# muestras]')
    plt.grid(True, which='both', ls=':', alpha=0.7)

    # Aplicar color de fondo a toda la figura
    plt.gcf().patch.set_facecolor(COLOR_FONDO)
    plt.tight_layout()
    plt.show()
    
    return b, retardo

def filtrar_FIR_ECG(b, nombre_filtro, ecg, fs, retardo):

    ecg_filt = signal.lfilter(b = b, a = 1, x = ecg)

    plt.figure()
    plt.plot(ecg, label='ecg crudo', alpha=0.7, color='forestgreen')
    plt.plot(ecg_filt, label=f'Filtrado ({nombre_filtro})', 
             linewidth=1.5, color=COLOR_ROSA)
    plt.title(f'ECG completo - {nombre_filtro}', fontweight='bold')
    plt.xlabel('Muestras (#)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True, alpha=0.7)
    # Color de fondo
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
        plt.plot(zoom_region, ecg_filt[zoom_region + retardo], label=nombre_filtro)

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
        plt.plot(zoom_region, ecg_filt[zoom_region + retardo], label=nombre_filtro)
       
        plt.title('ECG con ruido desde ' + str(ii[0]) + ' to ' + str(ii[1]) )
        plt.ylabel('Adimensional')
        plt.xlabel('Muestras (#)')
       
        axes_hdl = plt.gca()
        axes_hdl.legend()
        axes_hdl.set_yticks(())
               
        plt.show()
    
#para comparar ambos filtros fir
metodos_fir = ['firwin2', 'firls']
resultados_fir = {}

for metodo_actual in metodos_fir:
    coeficientes, retardo = filtro_FIR(
        fs=fs, 
        wp=wp, 
        ws=ws, 
        alpha_p=alpha_p, 
        alpha_s=alpha_s, 
        metodo=metodo_actual
    )
    resultados_fir[metodo_actual] = (coeficientes, retardo)

#filtro cada metodo
for nombre_metodo, (coef_filtro, retardo_temporal) in resultados_fir.items():
    filtrar_FIR_ECG(coef_filtro, nombre_metodo, ecg_one_lead, fs_ecg, retardo_temporal)