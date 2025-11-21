# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 17:01:38 2025

@author: gueva
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import scipy.io as sio

#%%
## ECG con ruido
fs = 1000 #[Hz]
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].squeeze()
N = len(ecg_one_lead)

#grafiquito normi
plt.figure()
plt.plot(ecg_one_lead[:50000], label='Original (ruidosa)', color='k', alpha=0.5, linewidth=1)
plt.legend()
plt.show()
#%%
#Si fs = 1000 Hz, significa que hay 1000 muestras por segundo → 
#es decir, 1 muestra = 1 ms.
ventana_200ms = int(0.2 * fs)   #0,2 segundos = 200 ms = 200 muestras
ventana_600ms = int(0.6 * fs)   #0,6 segundos = 600 ms = 600 muestras
#Esto convierte los tiempos de 200 ms y 600 ms en cantidad de muestras, 
#porque los filtros necesitan cantidad de puntos, no tiempo.

#corrijo por si no son impares
if ventana_200ms % 2 == 0:
    ventana_200ms += 1
if ventana_600ms % 2 == 0:
    ventana_600ms += 1

#primer filtrado-->elimina QRS y P 
med_1_200 = medfilt(ecg_one_lead, ventana_200ms)

#segundo filtrado
med_2_600 = medfilt(med_1_200, ventana_600ms)

#señal pipicucu
ecg_sig_final = ecg_one_lead - med_2_600

#graficulis
plt.figure(figsize=(12,4))
plt.plot(ecg_one_lead[:3000], label="Original", alpha=0.5)
plt.plot(ecg_sig_final[:3000], label="con filtro med")
plt.axhline(y=0, linestyle='--')
plt.legend()
plt.title("Filtro de Mediana 200 ms + 600 ms")
plt.show()

#%%
from scipy.interpolate import CubicSpline
#usamos flatten() para asegurar 1D
qrs = mat_struct['qrs_detections'].flatten()

#usamos directamente las posiciones QRS como puntos donde asumimos línea isoeléctrica
pto = qrs  
valor = ecg_one_lead[pto]  #valor de la señal en esos puntos

#interpolación con splines cúbicos 
n = np.arange(len(ecg_one_lead))  #tiempo total 
spline_baseline = CubicSpline(pto, valor)
baseline_spline = spline_baseline(n)

ecg_sig_final2 = ecg_one_lead - baseline_spline

#GRAFICOSSS!!
plt.figure(figsize=(12,4))
plt.plot(ecg_one_lead[:3000], label="ECG original", alpha=0.6)
plt.plot(ecg_sig_final2[:3000], label="ECG con interpolacion")
plt.axhline(y=0, linestyle='--')
plt.legend()
plt.title("ECG original vs corregido con spline cúbico)")
plt.show()

#%% 13/11 FILTRO ADAPTADO

patron = mat_struct['qrs_pattern1'].flatten()
patron_2 = patron - np.mean(patron)   # para tener area neta nula, util para filtrado

ecg_detection = signal.lfilter(b=patron_2, a=1, x=ecg_one_lead)

ecg_detection_abs = np.abs(ecg_detection)
ecg_detection_abs = ecg_detection_abs/np.std(ecg_detection_abs)

ecg_one_lead_dev = ecg_one_lead/np.std(ecg_one_lead)

mis_qrs, _ = signal.find_peaks(ecg_detection_abs, height=1, distance=300)  #300

plt.figure()
plt.plot(ecg_one_lead_dev)
plt.plot(ecg_detection_abs[57:])
plt.legend()

plt.show()

qrs_det = mat_struct['qrs_detections'].flatten()

def matriz_confusion_qrs(mis_qrs, qrs_det, tolerancia_ms=150, fs=1000):
    """
    Calcula matriz de confusión para detecciones QRS usando solo NumPy y SciPy
    
    Parámetros:
    - mis_qrs: array con tiempos de tus detecciones (muestras)
    - qrs_det: array con tiempos de referencia (muestras)  
    - tolerancia_ms: tolerancia en milisegundos (default 150ms)
    - fs: frecuencia de muestreo (default 360 Hz)
    """
    
    # Convertir a arrays numpy
    mis_qrs = np.array(mis_qrs)
    qrs_det = np.array(qrs_det)
    
    # Convertir tolerancia a muestras
    tolerancia_muestras = tolerancia_ms * fs / 1000
    
    # Inicializar contadores
    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives
    
    # Arrays para marcar detecciones ya emparejadas
    mis_qrs_emparejados = np.zeros(len(mis_qrs), dtype=bool)
    qrs_det_emparejados = np.zeros(len(qrs_det), dtype=bool)
    
    # Encontrar True Positives (detecciones que coinciden dentro de la tolerancia)
    for i, det in enumerate(mis_qrs):
        diferencias = np.abs(qrs_det - det)
        min_diff_idx = np.argmin(diferencias)
        min_diff = diferencias[min_diff_idx]
        
        if min_diff <= tolerancia_muestras and not qrs_det_emparejados[min_diff_idx]:
            TP += 1
            mis_qrs_emparejados[i] = True
            qrs_det_emparejados[min_diff_idx] = True
    
    # False Positives (tus detecciones no emparejadas)
    FP = np.sum(~mis_qrs_emparejados)
    
    # False Negatives (detecciones de referencia no emparejadas)
    FN = np.sum(~qrs_det_emparejados)
    
    # Construir matriz de confusión
    matriz = np.array([
        [TP, FP],
        [FN, 0]  # TN generalmente no aplica en detección de eventos
    ])
    
    return matriz, TP, FP, FN

# Ejemplo de uso

matriz, tp, fp, fn = matriz_confusion_qrs(mis_qrs, qrs_det)

print("Matriz de Confusión:")
print(f"           Predicho")
print(f"           Sí    No")
print(f"Real Sí:  [{tp:2d}   {fn:2d}]")
print(f"Real No:  [{fp:2d}    - ]")
print(f"\nTP: {tp}, FP: {fp}, FN: {fn}")

# Calcular métricas de performance
if tp + fp > 0:
    precision = tp / (tp + fp)
else:
    precision = 0

if tp + fn > 0:
    recall = tp / (tp + fn)
else:
    recall = 0

if precision + recall > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0

print(f"\nMétricas:")
print(f"Precisión: {precision:.3f}")
print(f"Sensibilidad: {recall:.3f}")
print(f"F1-score: {f1_score:.3f}")