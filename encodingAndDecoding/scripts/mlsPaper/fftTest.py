import pylab as plt
import numpy as np
import math

from numpy import fft


def buildUserData(data):
    data = data
    data = np.array(bitfield(data))
    # data_len = len(data)
    # data = np.repeat(data, codelength)
    # biPolarData = data*2-1
    # biPolarData = applyCarrier(biPolarData)
    # return (data, data_len)
    #print('data: ', data)
    return data


def bitfield(n):
    return [int(digit) for digit in bin(n)[2:]]


def getBitStrimForCompositeSignal(compositeSignal):
    bitStream = np.empty([1], dtype=int)
    for bit in compositeSignal:
        bitStream = np.append(bitStream, np.array(bitfield(bit)))
    print(bitStream)
    return bitStream


samplingFrequency = 320


def applyCarrier(data):
    bitWiseCarrier = []
    N = len(data)  # Number of symbols to be sent.
    N = 16  # Number of symbols to be sent.
    Fc = 100  # Carrier frequency.
    Fs = 350  # Sampling frequency.
    tStep = 1 / Fs  # Width of each symbol (in sec).

    t = np.linspace(0, (N - 1) * tStep, N)
    carrier = 2 * np.cos(2 * np.pi * Fc * t)
    for i in range(len(data)):
        modData = np.multiply(data[i], carrier)
        bitWiseCarrier.append(modData)

    # data = np.multiply(data, carrier)
    return bitWiseCarrier
    # return data


def continuousFFT(signals):
    fftBin = []
    for signal in signals:
        fftBin.append(applyfft(signal))
    return fftBin


def applyfft(signal):
    # print('signal: ', signal)
    Fs = samplingFrequency
    N = len(signal)
    N = 16
    fStep = Fs / N
    tStep = 1 / Fs
    f0 = 100
    f = np.linspace(0, (N - 1) * fStep, N)
    t = np.linspace(0, (N - 1) * tStep, N)
    y = 2 * np.cos(2 * np.pi * f0 * t)
    # X = np.fft.fft(signal, N)
    X = np.fft.fft(signal)
    count = 0
    for val in X:
        # print('index: ', count, ' val: ', val)
        count = count + 1
    ValArray = []
    for v in X:
        ValArray.append(math.sqrt(v.real ** 2 + v.imag ** 2))
    # print('FFT: ', ValArray)
    Xmag = np.abs(X) / N
    # Xmag = np.abs(X)

    fRealized = f[0: int(N / 2 + 1)]
    # print("frequency: ", fRealized)
    XmagRealized = 2 * Xmag[0: int(N / 2 + 1)]
    # XmagRealized = 2 * Xmag
    XmagRealized[0] = Xmag[0] / 2
    # print('FFT Mag: ', XmagRealized)

    # fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
    # ax1.plot(fRealized, XmagRealized, '.-')
    plt.plot(fRealized, XmagRealized, '.-')
    # ax2.plot(t, signal, '.-')
    plt.ylim([0, 10])
    plt.show()
    return np.amax(XmagRealized)


def spreadUserData(data, data_len, goldCode):
    data_code = []
    for i in range(data_len):
        data_code = np.append(data_code, goldCode)

    data_spread = np.logical_xor(data_code, data).astype(int)
    return (data_code, data_spread)



data1 = buildUserData(0xA)
data2 = buildUserData(0xB)
data1 = applyCarrier(data1)
data2 = applyCarrier(data2)

a = buildUserData(0xc1)
b = buildUserData(0xa5)
c = buildUserData(0xb5)
d = buildUserData(0xd5)
e = buildUserData(0xe5)

composite = (a) + (b) + (c) + d + (e)
print('composite= ', composite)

data = []
for i in range(len(data1)):
    l1 = data1[i]
    l2 = data2[i]
    data.append(l1 + l2)

print('Data1: ', data1)
print('Data: ', data)

# compositeBitStream = getBitStrimForCompositeSignal(data)
# modData = applyCarrier(compositeBitStream)
# modData = applyCarrier([3])
# modData = applyCarrier([1])
# modData = applyCarrier(data)
# applyfft(modData)

print(continuousFFT(data))
