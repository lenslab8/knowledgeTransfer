#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from comm import Modulador, Demodulador, Canal

# Configuration.

N = 100  # Number of symbols to be sent.
Fc = 100  # Carrier frequency.
Fs = 4 * Fc  # Sampling frequency.
Tb = 0.1  # Width of each symbol (in sec).


SNRdB = 100  # Signal power is twice the noise power.
SNR = 10.0 ** (SNRdB/10.0)


Fd = 5   # Doppler frequency for the multipath channel.

TAPS = 5  # Number of elements (paths) of the channel.

# Create the modulator and demodulator.
modulador = Modulador(Fc, Fs, Tb)
demodulador = Demodulador(modulador)
canal = Canal(SNR, TAPS, Fd)

# Data to be sent.
dados = np.random.choice(np.array([0, 1]), size=(N))

# Creating symbols for BPSK (-1 and 1).
simbolos_enviados = 2*dados - 1

# Modulates the symbols.
(ondaq_enviada, sinalm) = modulador.processar(simbolos_enviados)

# Process by channel.
sinalc = canal.processar(sinalm)

# Demodulates the received signal.
(sinald,  sinali, ondaq_recebida, simbolos_recebidos) = demodulador.processar(sinalc)

dados_recebidos = ((simbolos_recebidos + 1)/2).astype(int)

# Calculating decision errors.
num_erros = np.sum(simbolos_enviados != simbolos_recebidos)
BER = num_erros/simbolos_enviados.size

print('Do total de {} bits, {} foram decodificados de formada errada.'.format(
    simbolos_enviados.size, num_erros
))
print('BER: {}'.format(BER))

# Displaying transmitter graphs.
f1, (f1_ax1, f1_ax2, f1_ax3) = plt.subplots(3)
f1.suptitle('Sinal enviado a partir do transmissor', fontsize=14)
f1_ax1.stem(dados)
f1_ax1.set_title('Bits enviados')
f1_ax2.plot(ondaq_enviada)
f1_ax2.set_title('Onda quadrada gerada a partir dos símbolos')
f1_ax3.plot(sinalm)
f1_ax3.set_title('Sinal modulado')
f1.subplots_adjust(hspace=1)

# Displaying receiver graphics.
f2, (f2_ax1, f2_ax2, f2_ax3, f2_ax4, f2_ax5) = plt.subplots(5)
f2.suptitle('Sinal recebido no receptor.', fontsize=14)
f2_ax1.plot(sinalc)
f2_ax1.set_title('Sinal recebido do canal')
f2_ax2.plot(sinald)
f2_ax2.set_title('Sinal demodulado')
f2_ax3.plot(sinali)
f2_ax3.set_title('Sinal após integração')
f2_ax4.plot(ondaq_recebida)
f2_ax4.set_title('Onda quadrada recebida')
f2_ax5.stem(dados_recebidos)
f2_ax5.set_title('Dados recebidos')
f2.subplots_adjust(hspace=1)
plt.show()