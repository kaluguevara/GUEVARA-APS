# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:56:16 2025

@author: gueva
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ===================================
#Ejecicio TP0
def funcion_seno(amp, desplazo, frec, fase, cant_muestras, frec_muestreo):
    # Crear el vector de tiempos
    tt = (np.arange(cant_muestras)/frec_muestreo).reshape(-1, 1)
    # Calcular la señal senoidal
    xx = (amp * np.sin(2 * np.pi * frec * tt + fase) + desplazo).reshape(-1, 1)
    return tt, xx
"""
reshape asegura que los vectores tt y xx tengan forma (Nx1)
que es lo que pide la consigna. Tambien se puede hacer con np.linspace. 
Segun el diccionario de numpy, el -1 calcula el tamaño de la dimension, 
en este caso, nro de muestras. Y el 1, fija la segunda dimension en 1, 
o sea N filas x 1 columna.
"""
    # Generar señal
tt, xx = funcion_seno(amp=1, desplazo=0.5, frec=5, fase=np.pi/4, cant_muestras=1000, frec_muestreo=1000)
    
    # Graficar
plt.figure(1)
plt.plot(tt, xx)
plt.title('Funcion seno')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.grid(True)
plt.show()

# ==========================================
#Bonus: Experimento con frecuencias vistas en clase
lista_frec_experimento = [500,999,1001,2001]
plt.figure(2)
for i in range(len(lista_frec_experimento)):
    frec = lista_frec_experimento[i]
    tt, xx = funcion_seno(amp=1, desplazo=0.5, frec=frec, fase=np.pi/4, cant_muestras=1000, frec_muestreo=1000)
    plt.subplot(2, 2, i+1)  
    #plt.subplot(2 filas, 2 columnas, 1 a 4 plots.
    #al i le sumo 1 porque el subplot siempre arranca en 1
    plt.plot(tt, xx)
    plt.title(f'Funcion seno con frecuencia: {frec} Hz')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud (V)')
    plt.grid(True)

plt.tight_layout()
#ajusta automáticamente los espacios entre subplots para que no haya solapamiento o cortes,
#basicamente para que no quede feo
plt.show()

# ==========================================
#Bonus: Experimentar con otra funcion, elegi sawtooth, solo cambia que 
#agrego un argumento que es el ancho de los dientes de sierra en la funcion sawtooth
def generar_sawtooth(amp, desplazo, frec, fase, cant_muestras, frec_muestreo, width):
    # Vector de tiempos con arange
    tt = (np.arange(cant_muestras)/frec_muestreo).reshape(-1, 1)
    # Generar señal sawtooth
    xx = (amp * signal.sawtooth(2 * np.pi * frec * tt + fase, width=width) + desplazo).reshape(-1, 1)
    return tt, xx

    # Generar señal
tt, xx = generar_sawtooth(amp=1, desplazo=0.5, frec=5, fase=np.pi/4, cant_muestras=1000, frec_muestreo=1000, width=0.5)
     
    # Graficar
plt.figure(3)
plt.plot(tt, xx)
plt.title('Funcion sawtooth (diente de sierra)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.grid(True)
plt.show()

# ejemplo de uso
suma=(1/1000)*np.sum(xx**2)
energia=np.var(xx)

#autocorrelacion te exijen que k este invertida en tiempo, la invertis y desplazas, 
#pero esto es x*h, cuando es x*x solo desplazas
#porque autocorrelacion se relaciona con convolucion 