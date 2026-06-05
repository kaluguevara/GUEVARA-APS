from scipy import signal as sig
import numpy as np
import matplotlib.pyplot as plt

COLOR_ROSA = '#ff4fa3'
fs = 500
wp = 70
ws = 100
gpass = 3
gstop = 10

b_coeffs, a_coeffs = sig.iirdesign(wp, ws, gpass, gstop, analog=False, ftype='butter', output='sos ', fs=fs)

omega, h = sig.freqz(b_coeffs, a=a_coeffs, worN=1024, fs=fs)
fase = np.unwrap(np.angle(h))

zeros, poles, k = sig.tf2zpk(b_coeffs, a_coeffs)

plt.figure(figsize=(8,8))
plt.subplot(3,1,1)
plt.plot(omega, 20*np.log10(np.maximum(abs(h), 1e-10)), label='Butterworth', color=COLOR_ROSA, linewidth=2)

#plantilla

# Banda de paso
plt.fill([0, wp, wp, 0], [-gpass, -gpass, 5, 5], alpha=0.25, hatch='//', edgecolor='black', facecolor='lightgray', label='Plantilla')

# Banda de rechazo
plt.fill([ws, fs/2, fs/2, ws], [0, 0, -gstop, -gstop], alpha=0.25, hatch='//', edgecolor='black', facecolor='lightgray')

# Líneas verticales
plt.axvline(wp, color='black', linestyle='--')
plt.axvline(ws, color='black', linestyle='--')
plt.title('Respuesta en Magnitud', fontweight='bold')
plt.ylabel('|H(jω)| [dB]')
plt.xlabel('Frecuencia [Hz]')
plt.xlim(0, fs/2)
plt.ylim(-60, 5)
plt.grid(True, which='both', ls=':', alpha=0.7)
plt.legend()

#fase
plt.subplot(3,1,2)
plt.plot(omega, np.degrees(fase), color=COLOR_ROSA, linewidth=2)
plt.title('Fase', fontweight='bold')
plt.ylabel('Fase [°]')
plt.xlabel('Frecuencia [Hz]')
plt.grid(True, which='both', ls=':', alpha=0.7)

# polos y ceros
plt.subplot(3,1,3)
theta = np.linspace(0, 2*np.pi, 500)
plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.7)
plt.plot(np.real(zeros), np.imag(zeros), 'ob', label='Ceros')
plt.plot(np.real(poles), np.imag(poles), 'xr', markersize=10, label='Polos')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Diagrama de Polos y Ceros', fontweight='bold')
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginaria')
plt.grid(True, which='both', ls=':', alpha=0.7)
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()