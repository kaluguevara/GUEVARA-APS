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

#Sus graficos
plt.plot(t_seno, x1, color="hotpink")
plt.title('Funcion seno base')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.xlim(0, 0.005)
plt.grid(True)
plt.show()

#Misma señal amplificada y desfazada en π/2
x2 = 2 * np.sin(2 * np.pi * frec * t_seno + np.pi/2)

#Sus graficos
plt.plot(t_seno, x2, color="hotpink")
plt.title('Señal amplificada y desfasada π/2')
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

#Sus graficos
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

#Sus graficos
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

#Sus graficos
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

#Sus graficos
plt.plot(t_pulso, x_pulso, color="hotpink")
plt.title('Pulso rectangular de 10 ms')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.ylim(-0.1, 1.1)  # Para ver mejor los bordes
plt.xlim(0, 0.02)    # Para ver la transición  
plt.grid(True)
plt.show()


#me hago un diccionario con todas mis funciones asi puedo hacer un for, me ahorro lineas de codigo
signals = {
    "Funcion base seno": x1,
    "Seno amplificado y desfasado": x2,
    "Modulada": x3,
    "Clippeada": x1_clippeada,
    "Cuadrada": x_cuadrada,
    "Pulso rectangular": x_pulso
}

def calc_potencia(x): 
    return np.mean(x**2)

def calc_energia(x, Ts): 
    return np.sum(x**2) * Ts

#A señales periódicas le calculo la potencia y a las señales de duración finita la energía

print("Energía, potencia y número de muestras de cada señal (Nro_m):")
for name, sig in signals.items():
    if (name == "Pulso rectangular"): #!= devuelve un booleano en numpy
        Pot = calc_potencia(sig)
        print(f"{name}: su potencia es {Pot} y tiene {len(sig)} muestras.")
    else:
        E_pulso = calc_energia(x_pulso, Ts)
        print(f"{name}: su energia es {E_pulso} y tiene {len(sig)} muestras.")
        
señales = {
    "Seno amplificado y desfasado": x2,
    "Modulada": x3,
    "Clippeada": x1_clippeada,
    "Cuadrada": x_cuadrada[:len(t_seno)]  

}

print("Ortogonalidad entre x1 y las demás señales")
for nombre, sig in señales.items():
    sig_recortada = sig[:len(x1)]
    producto = np.dot(x1, sig_recortada)
    
    if np.isclose(producto, 0, atol=1e-6):
        resultado = "Es ortogonal"
    else:
        resultado = "No es ortogonal"
    print("Funcion base seno producto con", nombre, "=", f"{producto:.6f}", "-->", resultado)

#autocorrelación de x1
autoco_x1 = np.correlate(x1, x1, mode='full')

# correlación de x1 con otras señales
correlaciones = {}
for nombre, sig in señales.items():
    sig_recortada = sig[:len(x1)]  # recortamos si es necesario
    correlacion = np.correlate(x1, sig_recortada, mode='full')
    correlaciones[nombre] = correlacion

lags = np.arange(-len(x1)+1, len(x1))  

# autocorrelación
plt.plot(lags, autoco_x1, color="hotpink")
plt.title("Autocorrelación de x1")
plt.xlabel("Lag")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# correlación con otras señales
for nombre, correlacion in correlaciones.items():
    plt.plot(lags, correlacion, label=nombre)
plt.title("Correlación de funcion base seno con otras señales")
plt.xlabel("Lag")
plt.ylabel("Valor")
plt.grid(True)
plt.legend()
plt.show()

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
