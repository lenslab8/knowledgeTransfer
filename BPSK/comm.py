import numpy as np
from scipy.stats import rayleigh

# Modulator configuration.r.


class Modulador:

    def __init__(self, Fc, Fs, Tb):
        self.Fc = Fc
        self.Fs = Fs
        self.Ts = 1/Fs
        self.Tb = Tb  # Width of each symbol.
        self.L = int(Tb * Fs)  # Number of samples for the time of each symbol.

    def processar(self, simbolos):
        # Generate the square wave (each Tb lasts for L samples).
        ondaq = np.repeat(simbolos, self.L)

        # Generate the carrier.
        t = np.linspace(0, ondaq.size, ondaq.size)
        portadora = np.cos(2 * np.pi * self.Fc * t)
        sinalm = np.multiply(ondaq, portadora)

        return (ondaq, sinalm)

   #composite1 =  sinalm1 + sinalm2 + sinalm3 + sinalm4 + sinalm5 (same frequency) (same subcarrier but different nodes)
#composite2 =  sinalm1 + sinalm2 + sinalm3 + sinalm4 + sinalm5 (same frequency) (same subcarrier but different nodes)

   #if I need to get rid of carrier before fft

   #fft(composite1 + composite2)


class Demodulador:

    def __init__(self, modulador):
        self.modulador = modulador

    def processar(self, sinalm):

        # Stage 1

        # Generate the carrier.
        t = np.linspace(0, sinalm.size, sinalm.size)
        portadora = np.cos(2 * np.pi * self.modulador.Fc * t)
        sinald = np.multiply(sinalm, portadora)

        # Integra to improve waveform (optional).
        sinali = np.convolve(sinald, np.ones(self.modulador.L))

        # Remove delay from self.modulator.L - 1 samples
        sinali = sinali[int(self.modulador.L) - 1::]

        # Stage 2 (decision maker)

        # Decide whether it is 1 or -1 based on threshold 0.
        positivos = (sinali > 0)
        negativos = np.logical_not(positivos)

        ondaq = np.empty(sinali.size)
        ondaq[positivos] = 1
        ondaq[negativos] = -1

        # Do a subsampling to get the symbols.
        simbolos = ondaq[::self.modulador.L]

        return (sinald, sinali, ondaq, simbolos)


class Canal:

    def __init__(self, SNR, taps, Fd):
        self.SNR = SNR
        self.taps = taps
        self.Fd = Fd

    def processar(self, sinal):
        # Applying white gaussian noise.
        potencia_sinal = np.sum(np.square(sinal))/sinal.size

        # Generating white gaussian noise (mean = 0, variance = awgn power).
        potencia_ruido = potencia_sinal / self.SNR
        desvio_padrao = np.sqrt(potencia_ruido)
        ruido_gaussiano = np.random.normal(0, desvio_padrao, sinal.size)

        # Applies noise to the signal.
        sinal_ruidoso = sinal + ruido_gaussiano

        # Attenuates the signal.
        return sinal_ruidoso