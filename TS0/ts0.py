# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:56:16 2025

@author: gueva
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#%%
#Ejecicio TP0
def funcion_seno(amp, desplazo, frec, fase, cant_muestras, frec_muestreo):
    # Crear el vector de tiempos
    tt = (np.arange(cant_muestras)/frec_muestreo).reshape(-1, 1)
    
    # Calcular la señal senoidal
    xx = (amp * np.sin(2 * np.pi * frec * tt + fase) + desplazo).reshape(-1, 1)
    return tt, xx
    
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

#%%
#Bonus: Experimento con frecuencias vistas en clase:
lista_frec_experimento = [500,999,1001,2001]
plt.figure(2)
for i in range(len(lista_frec_experimento)):
    frec = lista_frec_experimento[i]
    if (frec == 500):
        tt, xx = funcion_seno(amp=1, desplazo=0.5, frec=frec, fase=np.pi/4, cant_muestras=1000, frec_muestreo=25000)
        plt.subplot(2, 2, i+1)  
        plt.plot(tt, xx)
        plt.title(f'Funcion seno con frecuencia:{frec}Hz')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud (V)')
        plt.grid(True)
    else:
        tt, xx = funcion_seno(amp=1, desplazo=0.5, frec=frec, fase=np.pi/4, cant_muestras=1000, frec_muestreo=1000)
        plt.subplot(2, 2, i+1)  
        plt.plot(tt, xx)
        plt.title(f'Funcion seno con frecuencia:{frec}Hz')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud (V)')
        plt.grid(True)

plt.tight_layout()
plt.show()

#%%
#Bonus: Experimentar con otra funcion:
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

#%%
#potencia de señal
vmax=np.sqrt(2)
tt, xx = funcion_seno(vmax, 0.5, 5, np.pi/4, 1000, 1000)
Px = np.var(xx)

SNR = 20

Pr = 1/(10**(SNR/10)) #en clase lo puse disitno estaba mal

#agrego esto porque sino me tira error, falta variables en random normal
mu = 0
n = len(xx)  

#el ruido
U_n = np.random.normal(mu, np.sqrt(Pr), n).reshape(-1,1)
#le puse reshape porque me tiraba error con las dimensiones

#señal mas ruido
xxn = xx + U_n

#tuqui grafico
plt.figure(4)
plt.clf()
plt.plot(tt, xxn, label='Señal con ruido')
plt.plot(tt, xx, 'r', lw=2, label='Señal original')
plt.title(f'f0 = {frec} Hz')
plt.legend()
plt.grid(True)
plt.show()

#%%
from scipy import signal as sig
n0 = 300   #muestras

dd = np.zeros(n0+1)
dd[n0] = 1

yy = sig.convolve(xx.flatten(), dd) #tmb me tiraba error con dimensiones

plt.figure(5)
plt.clf()
plt.plot(yy)


#%%
yy = 1/n * sig.convolve(U_n, np.flip(U_n))

plt.figure(6)
plt.clf()
plt.plot(yy)


#%%
B = 3  # bits
Vfs = 3  # Volts

qq = Vfs / 2**B
xxq = np.round(xx/qq)
#el error es e=xx-xxq, pero deberia multiplicar por qq para pasar de los niveles a volts devuelta
ee = (xx-xxq)*qq
plt.figure(7)
plt.hist(ee)
plt.title('error de quant.')
plt.show()