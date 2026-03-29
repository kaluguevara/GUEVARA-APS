import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square

#Parametros
frec = 2000 #frecuencia en Hz de la señal principal
frec_muestreo = 50000  #frecuencia de muestreo [Hz]
Ts = 1/frec_muestreo   #tiempo entre muestras

# Vectores de tiempo usando arange
t_seno = np.arange(0, 0.005, Ts)       
t_cuadrada = np.arange(0, 0.005, Ts)   
t_pulso = np.arange(0, 0.02, Ts)        

#Una señal sinusoidal de 2KHz.
x1 = np.sin(2 * np.pi * frec * t_seno)
T1 = 1 / frec
N1 = len(t_seno)

print("1) Seno 2 kHz")
print("Periodo:", T1)
print("Muestras:", N1)
print("Tipo: Señal de potencia\n")

#Sus graficos
plt.figure(1)
plt.plot(t_seno, x1, color="hotpink")
plt.title('Seno 2 kHz')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.xlim(0, 0.005)
plt.grid(True)
plt.show()

#2)Misma señal amplificada 3 dB y desfasada en π/2.
#3dB = sqrt(2)
x2 = np.sqrt(2) * np.sin(2 * np.pi * frec * t_seno + np.pi/2)

print("2) Amplificada 3 dB + fase")
print("Periodo:", T1)
print("Muestras:", N1)
print("Tipo: Señal de potencia\n")

#Sus graficos
plt.figure(2)
plt.plot(t_seno, x2, color="hotpink")
plt.title('Amplificada 3 dB + fase')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.xlim(0, 0.005)  
plt.grid(True)
plt.show()

#Misma señal pero modulada en amplitud por otra de la mitad de frecuencia
frecmoduladora = 1000  # señal moduladora
m = 0.6  # el índice m sirve para controlar que la amplitud no se vuelva negativa ni excesiva
#va de 0 a 1
x3 = (1 + m * np.sin(2 * np.pi * frecmoduladora * t_seno)) * x1
#portadora * (1 + moduladora), el 1 se suma para que la envolvente nunca cambie de signo.
#modulacion de amplitud clasica se escribe asi 
T3 = 1 / frecmoduladora

print("3) Modulada en amplitud")
print("Periodo moduladora:", T3)
print("Muestras:", N1)
print("Tipo: Señal de potencia\n")

#Sus graficos
plt.figure(3)
plt.plot(t_seno, x3, color="hotpink")
plt.title('Señal de 2 kHz modulada en amplitud por 1 kHz')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.xlim(0, 0.005)  
plt.grid(True)

#Dibujo la envolvente
envolvente_superior = 1 + m * np.sin(2 * np.pi * frecmoduladora * t_seno)
envolvente_inferior = -(1 + m * np.sin(2 * np.pi * frecmoduladora * t_seno))
plt.plot(t_seno, envolvente_superior, 'g--')
plt.plot(t_seno, envolvente_inferior, 'g--')
plt.show()

#Señal anterior recortada al 75% de su amplitud
A = np.max(np.abs(x1))   #valor max de amp o.g
Aclip = 0.75 * A        #le saco el 75%
x1_clippeada = np.clip(x1, -Aclip, Aclip)

print("4) Señal clippeada 75%")
print("Periodo:", T1)
print("Muestras:", N1)
print("Tipo: Señal de potencia\n")

#Sus graficos
plt.figure(4)
plt.plot(t_seno, x1_clippeada, color="hotpink")
plt.title('Señal anterior recortada al 75 % de su potencia')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.xlim(0, 0.005)  
plt.grid(True)
plt.show()

#Señal cuadrada de 4 kHz
frec_cuadrada=4000
amp = 1    
x_cuadrada = amp * square(2 * np.pi * frec_cuadrada * t_cuadrada)
T5 = 1 / frec_cuadrada
N5 = len(t_cuadrada)

print("5) Cuadrada 4 kHz")
print("Periodo:", T5)
print("Muestras:", N5)
print("Tipo: Señal de potencia\n")

#Sus graficos
plt.figure(5)
plt.plot(t_cuadrada, x_cuadrada, color="hotpink")
plt.title('Señal cuadrada de 4 kHz')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.xlim(0, 0.00125)  
plt.grid(True)
plt.show()

#para el pulso rectangular defino un tiempo más largo para que entre, porque si uso el que ya tenía definido se ve como el pulso de un muerto
pulso = 0.01
x_pulso = np.where(t_pulso < pulso, 1, 0)  # vale 1 de 0 a 10 ms, 0 después
N6 = len(t_pulso)

print("6) Pulso rectangular")
print("Duración:", pulso)
print("Muestras:", N6)
print("Tipo: Señal de energia\n")

#Sus graficos
plt.figure(6)
plt.plot(t_pulso, x_pulso, color="hotpink")
plt.title('Pulso rectangular de 10 ms')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.ylim(-0.1, 1.1)  # Para ver mejor los bordes
plt.xlim(0, 0.02)    # Para ver la transición  
plt.grid(True)
plt.show()

