# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 20:06:02 2025

@author: gueva
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as sig



def blackman_tukey(x, M):
    """
    estimación de la PSD usando Blackman–Tukey.
    x es la senal, M es el numero de retardos maximos 
    como estamos hablando de la autocorrelacion, el retardo es cuanto corro la senal antes de compararla con ella misma
    si es cero es prque estoy comparando la senal con ella msma, si es 1 comparo con una version desplazada una muestra y asi
    """
    #una senal de N puntos tiene 2N-1 retardos, usarlos todos mete mucho ruido entonces blackman tukey recorta y solo se queda  con 2M-1 valores de autocorrelacion
    #M chico implica un espectro borroso pero suave
    #m grande espectro detallado pero con mucho ruido
    N = len(x)
    
    #atc completa
    r = np.correlate(x, x, mode='full')
    mid = len(r)//2
    r = r[mid-M+1 : mid+M]   #recorto retardos
    r = r / N                #normalizo
    
    #aplico la ventana a la autocorrelacion
    r = r * sig.windows.blackman(len(r))
    
    #la fft de la autocorrelacion me da la densidad espectral de potencia
    Pxx = np.abs(np.fft.fft(r, n=N))
    return Pxx


#parametros generales
fs = 1000.0  # frecuencia de muestreo (Hz)
N = 1000     # cantidad de muestras

ts = 1/fs    # tiempo de muestreo
df = fs/N    # resolución espectral

cant_pad = 1  # cantidad de zero-padding



tt = np.linspace(0, (N-1)*ts, N)   # grilla temporal
ff = np.linspace(0, (N-1)*df, N)   # grilla frecuencial

f0 = np.array([N/4])   # frecuencias base

plt.close('all')

for ii in f0:
    
    #la senal son dos senoidales
    xx = np.sin(2*np.pi*(ii)*df*tt).reshape(N,1) + \
         np.sin(2*np.pi*(ii+5)*df*tt).reshape(N,1)
    
    # normalización a potencia unitaria
    xx = xx / np.sqrt(np.var(xx, axis=0))

    # zero padding
    zz = np.zeros_like(xx)
    xx_pad = np.vstack((xx, zz.repeat((cant_pad-1), axis=0)))
    N_pad = xx_pad.shape[0]    
    df_pad = fs/N_pad
    ff_pad = np.linspace(0, (N_pad-1)*df_pad, N_pad)

    xx_pad = xx_pad / np.sqrt(np.var(xx_pad, axis=0))
    
    #estimadores
    ft_XX_pdg = 1/N_pad * np.fft.fft(xx, axis=0)             # Periodograma
    ft_XX_bt = blackman_tukey(xx.ravel(), M=N//5)            # Blackman–Tukey
    ff_wl, ft_XX_wl = sig.welch(xx.ravel(), nperseg=N//5)    # Welch
    
    #máscara hasta fs/2
    bfrec = ff <= fs/2
    bfrec_pad = ff_pad <= fs/2
    
    #potencias totales
    xx_pot_pdg = np.sum(np.abs(ft_XX_pdg)**2, axis=0)
    xx_pot_bt  = np.sum(np.abs(ft_XX_bt)**2, axis=0)
    xx_pot_wl  = np.sum(np.abs(ft_XX_wl)**2, axis=0)
    
    #ventana duplicadora
    ww = np.vstack((1, 2*np.ones((N//2-1,1)) ,1))
    
  
    plt.figure()
    plt.plot(ff[bfrec], 10*np.log10(ww * np.abs(ft_XX_pdg[bfrec,:])**2 + 1e-10), 
             ls='dotted', marker='o', label=f'Per. σ²={xx_pot_pdg[0]:3.3}')
    plt.plot(ff[bfrec], 10*np.log10(np.abs(ft_XX_bt[bfrec])**2 + 1e-10), 
             ls='dotted', marker='o', label=f'BT σ²={xx_pot_bt:3.3}')
    plt.plot(ff_wl*fs, 10*np.log10(np.abs(ft_XX_wl)**2 + 1e-10), 
             ls='dotted', marker='o', label=f'Welch σ²={xx_pot_wl:3.3}')
    
    plt.ylabel('Densidad de Potencia [dB]')
    plt.xlabel('Frecuencia [Hz]')
    plt.title('PSD de una senoidal con diferentes desintonías')
    plt.legend()
    plt.show()
