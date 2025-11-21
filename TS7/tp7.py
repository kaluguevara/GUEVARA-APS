# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 23:12:31 2025

@author: gueva
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#copio el a b c de la consigna
#es casi igual al ts6
sistemas = {
    "a) y(n)=x(n-3)+x(n-2)+x(n-1)+x(n)": ([1, 1, 1, 1], [1]),
    "b) y(n)=x(n-4)+x(n-3)+x(n-2)+x(n-1)+x(n)": ([1, 1, 1, 1, 1], [1]),
    "c) y(n)=x(n)-x(n-1)": ([1, -1], [1]),
    "d) y(n)=x(n)-x(n-2)": ([1, 0, -1], [1]),
}

for nombre, (b, a) in sistemas.items():
    #rta en frec
    w, h = signal.freqz(b, a, worN=512)  
    mag = np.abs(h)
    fase = np.angle(h)

    #grafico modulo
    plt.figure(figsize=(10, 5))
    plt.plot(w, 20 * np.log10(mag + 1e-12), color='mediumvioletred', linewidth=2)
    plt.title(f"{nombre} - Módulo en dB")
    plt.xlabel('Frecuencia [rad/muestra]')
    plt.ylabel('|H(e^jω)| [dB]')
    plt.grid(True)
    plt.show()

    #grafico fase
    plt.figure(figsize=(10, 5))
    plt.plot(w, np.degrees(np.unwrap(fase)), color='deeppink', linewidth=2)
    plt.title(f"{nombre} - Fase")
    plt.xlabel('Frecuencia [rad/muestra]')
    plt.ylabel('Fase [°]')
    plt.grid(True)
    plt.show()

    #resultados
    print(f"\n{nombre}")
    print(f"  Módulo máximo = {np.max(mag):.3f}")
    print(f"  Fase final (último punto) = {np.degrees(fase[-1]):.2f}°")
