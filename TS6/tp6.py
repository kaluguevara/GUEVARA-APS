# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 15:33:22 2025

@author: gueva
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#funciones de transferencia, me armo un diccionario con las transferencias, 
#pongo los coeficientes que corresponden
sistemas = {
"T₁(s) = (s² + 9)/(s² + √2·s + 1)": ([1, 0, 9], [1, np.sqrt(2), 1]),
"T₂(s) = (s² + 1/9)/(s² + 0.2·s + 1)": ([1, 0, 1/9], [1, 0.2, 1]),
"T₃(s) = (s² + 0.2·s + 1)/(s² + √2·s + 1)": ([1, 0.2, 1], [1, np.sqrt(2), 1]),
}
for T, (b, a) in sistemas.items():
    
    #saco los polos y ceros
    z, p, k = signal.tf2zpk(b, a)
    
    #diagrama de polos y ceros
    plt.figure(figsize=(12, 4))
    plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label='Polos')
    
    #z es el arreglo de ceros como algunas funciones de transferencia pueden no tener ceros
    #hay que chequear si hay antes de plottear
    if len(z) > 0:
        plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label='Ceros')
    plt.title(f'{T} - Diagrama de Polos y Ceros')
    plt.xlabel('Parte Real (σ)', fontsize=11)
    plt.ylabel('Parte Imaginaria (jω)', fontsize=11)
    plt.axhline(0, color='gray', lw=1)
    plt.axvline(0, color='gray', lw=1)
    plt.grid(True, ls='--', color='lightgray')
    plt.legend(frameon=False)

    #Rta en frecuencia
    w, h = signal.freqs(b=b, a=a, worN=np.logspace(-1, 2, 1000))
    fase = np.unwrap(np.angle(h))
    #corrige esos saltos, sumando o restando múltiplos de 2π para que la fase quede continua y suave
    gd = -np.diff(fase) / np.diff(w)
    #calcula el retardo de grupo
    
    #magnitud
    plt.figure(figsize=(12, 4))
    plt.semilogx(w, 20 * np.log10(abs(h)), color='hotpink', linewidth=2)
    #el eje X se comprime logarítmicamente, mostrando mejor cómo cambia 
    #la fase en todo el rango de frecuencias.
    plt.title(f'{T} - Respuesta en Magnitud')
    plt.xlabel('Pulsación angular [rad/s]')
    plt.ylabel('|H(jω)| [dB]')
    plt.grid(True, which='both', ls='--')
    
    #fase
    plt.figure(figsize=(12, 4))
    plt.semilogx(w, np.degrees(fase), color='hotpink', linewidth=2)
    plt.title(f'{T} - Fase')
    plt.xlabel('Pulsación angular [rad/s]')
    plt.ylabel('Fase [°]')
    plt.grid(True, which='both', ls='--')
    
    plt.tight_layout()
    plt.show()
