# -*- coding: utf-8 -*-

"""Ajustando una señal electromiográfica funcional:

-Laboratorio Integrativo de Biomecánica y Fisiología del Esfuerzo,
Escuela de Kinesiología, Universidad de los Andes, Chile-
-Escuela de Ingeniería Biomédica, Universidad de Valparaíso, Chile-
        --Profesores: Oscar Valencia & Alejandro Weinstein--

"""
# Importar librerias
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def ajusta_emg_func(emg_fun, emg_cvm, fs, fc, filt_ord):
    """Ajusta EMG funcional según contracción voluntaria máxima.

    La función utiliza una señal EMG funcional y otra basada en la
    solicitación de una contracción isométrica voluntaria máxima. Ambas señales
    son procesadas considerando su centralización (eliminación de
    "offset"), rectificación y filtrado (pasa bajo con filtfilt).

    Parameters
    ----------
    emg_fun : array_like
        EMG funcional del músculo a evaluar
    emg_cvm : array_like
        EMG vinculada a la contracción voluntaria máxima del mismo músculo
    fs : float
       Frecuencia de muestreo, en hertz, de la señal EMG. Debe ser la misma
       para ambas señales.
    fc : float
        Frecuencia de corte, en hertz, del filtro pasa-bajos.
    filt_ord : int
        Orden del filtro pasa bajos

    Return
    ------
    emg_fun_norm : array_like
        EMG funcional filtrada y  normalizada
    emg_fun_env_f : array_like
        Envolvente de EMG funcional filtrada
    emg_cvm_envf_ : array_like
        Envolvente de EMG CVM filtrada
    """
    #centralizando y rectificando las señales EMG
    emg_fun_env = abs(emg_fun - np.mean(emg_fun))
    emg_cvm_env = abs(emg_cvm - np.mean(emg_cvm))

    # Filtrado pasa-bajo de las señales
    b, a = butter(int(filt_ord), (int(fc)/(fs/2)), btype = 'low')
    emg_fun_env_f = filtfilt(b, a, emg_fun_env)
    emg_cvm_env_f = filtfilt(b, a, emg_cvm_env)

    #calculando el valor máximo de emg_cvm y ajustando la señal EMG funcional
    emg_cvm_I = np.max(emg_cvm_env_f)
    emg_fun_norm = (emg_fun_env_f / emg_cvm_I) * 100
    
    return emg_fun_norm, emg_fun_env_f, emg_cvm_env_f


#%%

def plot_emgs(emg_fun, emg_fun_env, emg_fun_norm, emg_cvm, emg_cvm_env,
              fs, f_c, f_orden,
              nombre):
    """Grafica señales de EMG funcional y CVM.

    Parameters
    ----------
    emg_fun : array_like
        EMG funcional.
    emg_fun_env : array_like
        Envolvente del EMG funcional.
    emg_fun_norm : array_like
        EMG funcional normalizada según CVM.
    emg_cvm : array_like
        EMG contracción voluntaria máxima.
    fs : float
        Frecuencia de muestreo, en hertz.
    f_c : float
        Frecuencia de corte del filtro pasa-bajo, en hertz.
    f_orden : int
        Orden del filtro.
    nombre : str
        Nombre del músculo.
    """

    # Vectores de tiempo
    t1 = np.arange(0, len(emg_fun) / fs, 1 / fs)
    t2 = np.arange(0, len(emg_cvm) / fs, 1 / fs)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize = (8, 7))

    ax1.plot(t1, emg_fun, 'b', label='Señal bruta')
    ax1.set_title(f'Músculo: {nombre}; filtro aplicado: f_c={f_c} [Hz] y '
                  f'orden {f_orden}')

    ax1.plot(t1, emg_fun_env, 'r', lw=2, label='Señal filtrada')
    ax1.set_ylabel(f'{nombre} Funcional\nAmplitud [V]',fontsize=9)
    ax1.set_ylim(emg_fun.min() - 0.1, emg_fun.max() + 0.1)
    ax1.set_xlim(0, t1.max())
    ax1.grid()
    ax1.legend(loc='upper center', fontsize='x-small', borderpad=None)

    ax2.plot(t2, emg_cvm, 'b', label='Señal bruta')
    ax2.plot(t2, emg_cvm_env, 'r', lw=2, label='Señal filtrada')
    ax2.set_ylabel(f'{nombre} CVM\nAmplitud [V]',fontsize=9)
    ax2.axvline((np.argmax(emg_cvm_env) / fs), color='maroon')
    ax2.text(0.85, 0.95 ,f'Max = {emg_cvm_env.max():.2f}',
             transform=ax2.transAxes, ha="left", va="top")
    ax2.set_ylim(emg_cvm.min() - 0.1, emg_cvm.max() + 0.1)
    ax2.set_xlim(0, t2.max())
    ax2.grid()
    ax2.legend(loc='upper center', fontsize='x-small', borderpad=None)

    ax3.plot(t1, emg_fun_norm, 'g',label='Señal ajustada según CVM')
    ax3.set_ylim(emg_fun_norm.min(), emg_fun_norm.max() + 2)
    ax3.set_xlim(0, t1.max())
    ax3.set_xlabel('Tiempo [s]', fontsize=9)
    ax3.set_ylabel('% EMG CVM')
    ax3.grid()
    ax3.legend(loc='upper center', fontsize='x-small', borderpad=None)

    plt.tight_layout(h_pad=.1)
    
    
#%% ejemplo para utilizar funciones

if __name__ == '__main__':
    df_funcional = pd.read_csv('emg_funcional.csv')
    df_cvm = pd.read_csv('emg_cvm.csv')

    musculo = 'GM'
    emg_funcional = df_funcional[musculo].to_numpy()
    emg_cvm = df_cvm[musculo].to_numpy()
    fs = 1e3
    fc, forden = 40, 2
    emg_f_n, emg_f_env, emg_cvm_env = ajusta_emg_func(emg_funcional,
                                                      emg_cvm, fs, fc, forden)

    #imprime el valor máximo de la señal funcional ajustada y la emg_cvm
    print(f'Valor máximo de la señal CVM {emg_cvm_env.max():.2f} V')
    print(f'% de activación máxima de la señal ajustada:{emg_f_n.max():.2f}%')

    plt.close('all')
    plot_emgs(emg_funcional, emg_f_env, emg_f_n, emg_cvm, emg_cvm_env,
              fs, fc, forden, 'GM')
    plt.savefig('emg.png')
    plt.savefig('emg.pdf')
    plt.show()
