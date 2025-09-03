# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 11:45:41 2025

@author: gueva
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ===================================
    #Señal sinusoidal de 2 kHz
def funcion_seno(amp, desplazo, frec, fase, cant_muestras, frec_muestreo):
    # Crear el vector de tiempos
    tt = (np.arange(cant_muestras)/frec_muestreo).reshape(-1, 1)
    # Calcular la señal senoidal
    xx = (amp * np.sin(2 * np.pi * frec * tt + fase) + desplazo).reshape(-1, 1)
    return tt, xx

    # Generar señal
tt, xx = funcion_seno(amp=1, desplazo=0, frec=2000, fase=0, cant_muestras=1000, frec_muestreo=50000)
#Frecuencia de muestreo al menos 10 veces la frecuencia máxima para evitar aliasing
#Potencia: si la amplitud es 1, la potencia promedio es P=A^2/2=0,5

    # Graficar
plt.subplot(2, 3, 1)
plt.plot(tt, xx)
plt.title('Funcion seno base')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.xlim(0, 0.005)  # ← ¡ESTO HACE EL ZOOM!
plt.grid(True)
plt.show()

# ===================================
    #Señal amplificada y desfasada π/2
tt, xx2 = funcion_seno(amp=2, desplazo=0, frec=2000, fase=np.pi/2, cant_muestras=1000, frec_muestreo=50000)
plt.subplot(2, 3, 2)
plt.plot(tt, xx2)
plt.title('Señal amplificada y desfasada π/2')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.xlim(0, 0.005)  # ← ¡ESTO HACE EL ZOOM!
plt.grid(True)
plt.show()

# ===================================
    #Señal modulada en amplitud por otra señal senoidal de 1 kHz
    #Para modular podriamos multiplicar por un seno con frec 1000 o modular poniendo un seno en la amplitud del seno base
tt, mod = funcion_seno(amp=1, desplazo=0, frec=1000, fase=0, cant_muestras=1000, frec_muestreo=50000)   
xx3 = (mod.flatten()) * (xx.flatten())
""" tenemos un problema, no puedo multiplicar dos senos a la zarasa xq es un arreglo multidimensional, 
segun el reshapee (N,1). Entonces tengo que convertirlo en 1 dimension e ir multiplicando termino a 
termino los xx, eso se le dice aplanar o flatten. agarro el seno base y con el que quiero modular, 
los flatteneo y multiplico
"""

plt.subplot(2, 3, 3)
plt.plot(tt, xx3)
plt.title('Señal de 2 kHz modulada en amplitud por 1 kHz')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.xlim(0, 0.005)  # ← ¡ESTO HACE EL ZOOM!
plt.grid(True)
plt.show()

# ===================================
    #Señal anterior recortada al 75 % de su potencia--> CLIPPING
def aplicar_clipping(senal, porcentaje_amplitud):
    '''
    senal: señal a recortar
    porcentaje_amplitud: Porcentaje de la amplitud a mantener (0-100)
    '''
    # Convertir porcentaje a fracción
    fraccion = porcentaje_amplitud / 100.0
    
    # Calcular límites basados en la amplitud máxima de la señal
    amp_max = np.max(np.abs(senal))
    limite_superior = fraccion * amp_max
    limite_inferior = -fraccion * amp_max
    
    # Aplicar clipping
    senal_recortada = np.copy(senal)
    senal_recortada[senal_recortada > limite_superior] = limite_superior
    senal_recortada[senal_recortada < limite_inferior] = limite_inferior
    
    return senal_recortada

# Usar la función
xx4 = aplicar_clipping(xx, 75)  # Clipping al 75%

plt.subplot(2, 3, 4)
plt.plot(tt, xx4)
plt.title('Señal anterior recortada al 75 % de su potencia')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.xlim(0, 0.005)  # ← ¡ESTO HACE EL ZOOM!
plt.grid(True)
plt.show()


# ===================================
    #Señal cuadrada de 4 kHz
def generar_senal_cuadrada(amp, desplazo, frec, fase, cant_muestras, frec_muestreo):
    tt = (np.arange(cant_muestras)/frec_muestreo).reshape(-1, 1)
    xx = signal.square(2*np.pi*frec*tt).reshape(-1,1)
    return tt, xx

tt, xx5 = generar_senal_cuadrada(1, 0, 4000, 0, 1000, 50000)

plt.subplot(2, 3, 5)
plt.plot(tt, xx5)
plt.title('Señal cuadrada de 4 kHz')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.xlim(0, 0.005)  # ← ¡ESTO HACE EL ZOOM!
plt.grid(True)
plt.show()

# ===================================
    #Pulso rectangular de 10 ms
    
def generar_pulsorect(T_pulso, frec_muestreo, duracion_total, amplitud):

    # Muestras totales
    N_total = int(frec_muestreo * duracion_total)
    N_pulso = int(frec_muestreo * T_pulso)
    
    # Crear vector de tiempo completo
    tt = (np.arange(N_total) / frec_muestreo).reshape(-1, 1)
    
    # Crear pulso rectangular: 1 durante T_pulso, 0 en otro caso
    xx = np.zeros_like(tt)
    xx[:N_pulso] = amplitud  # Pulso al inicio
    
    return tt, xx

# Generar pulso de 10 ms con señal de 20 ms de duración total
tt, xx6 = generar_pulsorect(0.01, 50000, 0.02, 1)

# Graficar
plt.subplot(2, 3, 6)
plt.plot(tt, xx6, 'b-', linewidth=2)
plt.title('Pulso rectangular de 10 ms')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)  # Para ver mejor los bordes
plt.xlim(0, 0.02)    # Para ver la transición
plt.show()



# ===================================
#En cada caso indique tiempo entre muestras, número de muestras y potencia o energía según corresponda.
#HAGO UN PRINT???????????/
# ===================================
senales = {
    "ampl+desfase π/2": xx2,
    "AM 1 kHz":          xx3,
    "75% de potencia":   xx4,
    "cuadrada 4 kHz":    xx5,
    "pulso 10 ms":       xx6
}
def es_ortogonal(a, b):
    atol = 1e-10
    # Asegurar vectores 1D
    a = a.flatten()
    b = b.flatten()
    # producto interno discreto
    productointerno = np.dot(a, b)
    ortogonalidad = np.isclose(productointerno, 0.0, atol) #esto es un booleano, true o fal
    return productointerno, ortogonalidad

for nombre, y in senales.items():
    prod, orto = es_ortogonal(xx, y)
    if orto:
        print(f"{nombre:18s} -> producto interno = {prod:.4e} -> Son ortogonales")
    else:
        print(f"{nombre:18s} -> producto interno = {prod:.4e} -> No son ortogonales")

# ===================================
#autocorrelacion
def correlacion(x, y):
    # Asegurar vectores 1D
    x = x.flatten()
    y = y.flatten()
    # np.correlate con mode='full' devuelve longitudes 2N-1
    correlacion = np.correlate(x, y, mode='full')
    return correlacion

# --- autocorrelación de la señal base ---
auto = correlacion(xx, xx)
lags_auto = np.arange(-len(xx)+1, len(xx))

plt.figure()
plt.plot(lags_auto, auto)
plt.title('Autocorrelación de la señal base (2 kHz)')
plt.xlabel('Desplazamiento (muestras)')
plt.ylabel('Valor de autocorrelación')
plt.grid(True)
plt.show()

# --- correlación cruzada entre señal base y las demás ---
for nombre, y in senales.items():
    corr = correlacion(xx, y)
    lags = np.arange(-len(xx)+1, len(xx))  # mismo eje
    plt.figure()
    plt.plot(lags, corr)
    plt.title(f'Correlación entre señal base y "{nombre}"')
    plt.xlabel('Desplazamiento (muestras)')
    plt.ylabel('Valor de correlación')
    plt.grid(True)
    plt.show()

# ===================================
# VERFICAR IGUALDAD

frecuencia = 200  
w = 2 * np.pi * frecuencia  # Frecuencia angular ω
duracion = 0.01  # 10 ms de señal
frec_muestreo = 44100  # Frecuencia de muestreo
cant_muestras = int(duracion * frec_muestreo)

# Crear vector de tiempo
t = (np.arange(cant_muestras) / frec_muestreo).reshape(-1, 1)

alpha = w * t
beta = 2 * alpha 

# Calcular ambos lados de la identidad
ecuacion1= 2 * np.sin(alpha) * np.sin(beta)
ecuacion2 = np.cos(alpha - beta) - np.cos(alpha + beta)

# Verificar la igualdad (con tolerancia para errores numéricos)
if np.allclose(ecuacion1, ecuacion2, atol=1e-10):
    print("La identidad trigonométrica se cumple!")
else:
    print("La identidad no se cumple :(")

# ===================================
# BONNUSSSSSSSSSSSSSSSSSS!

import scipy.io.wavfile as wavfile

def analizar_audio(archivo_wav):
    
    # Leer el archivo WAV
    frecuencia_muestreo, datos_audio = wavfile.read(archivo_wav)
    
    # Convertir a mono si es estéreo
    '''
    Mono: Una sola señal de audio (1 canal). Todos los altavoces reproducen el mismo sonido
    estereo: Dos señales de audio (2 canales: izquierdo y derecho). Sonido espacial con dirección y profundidad
    ¿Por qué convertir a mono para análisis?
    - Un solo canal es más fácil de analizar
    - Evita duplicar energía al sumar ambos canales
    '''
    #--> caro pedile al chat que te explique pero es bardo
    if len(datos_audio.shape) > 1:
        datos_audio = datos_audio.mean(axis=1)
        print("Convertido a mono")
    
    # Normalizar entre -1 y 1 --> caro pedile al chat que te explique pero es bardo
    datos_normalizados = datos_audio.astype(np.float32) / np.max(np.abs(datos_audio))
    # si no normalizo son valores muy grandes
    
    # Calcular energía
    energia = np.sum(datos_normalizados**2)
    print(f"Energía total: {energia:.4f}")
    
    # Crear vector de tiempo
    tiempo = np.arange(len(datos_normalizados)) / frecuencia_muestreo
    
    # Graficar
    plt.figure(figsize=(12, 6))
    plt.plot(tiempo, datos_normalizados)
    plt.title('Bonus: señal de audio de freesound.org')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud normalizada')
    plt.grid(True)
    plt.show()
    
    return datos_normalizados, frecuencia_muestreo, energia

# USO DEL CÓDIGO
archivo = "soundwav.wav"  # CAMBIA POR EL NOMBRE DE TU ARCHIVO
datos, fs, energia = analizar_audio(archivo)
