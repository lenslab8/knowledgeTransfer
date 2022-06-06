import math

import numpy as np
import pylab as plt
from numpy import fft

import gold


def cconv(x, y):
    return fft.ifft(fft.fft(x) * fft.fft(y))


def ccorr(x, y):
    return fft.ifft(fft.fft(x) * fft.fft(y).conj())


def getAccuracy(sentData, reveivedData, threshold):
    sentData = sentData * 2 - 1
    success = 0
    fail = 0
    for s, r in zip(sentData, reveivedData):
        # r = 1 if abs(r) > 1 else abs(r)
        # accuracy = abs(abs(s) - abs(r))
        if (abs(r) >= threshold):
            success = success + 1
        else:
            # print('s: ', s, ' r:', r)
            fail = fail + 1

    # print('fail: ', fail)
    # print('success: ', success)
    return success / len(sentData) * 100


def bitfield(n):
    return [int(digit) for digit in bin(n)[2:]]


def buildUserData(data, codelength):
    data = data
    data = np.array(bitfield(data))
    data_len = len(data)
    data = np.repeat(data, codelength)
    # biPolarData = data*2-1
    # biPolarData = applyCarrier(biPolarData)
    return (data, data_len)


def spreadUserData(data, data_len, goldCode):
    data_code = []
    for i in range(data_len):
        data_code = np.append(data_code, goldCode)

    data_spread = np.logical_xor(data_code, data).astype(int)
    # data_spread = applyCarrier(data_spread*2-1)
    # data_spread = applyCarrier(data_spread)
    data_spread = applyCarrierModified(data_spread, 100)

    return (data_code, data_spread)


def spreadUserData2(data, data_len, goldCode):
    data_code = []
    for i in range(data_len):
        data_code = np.append(data_code, goldCode)

    data_spread = np.logical_xor(data_code, data).astype(int)
    # data_spread = applyCarrier2(data_spread*2-1)
    data_spread = applyCarrier2(data_spread)

    return (data_code, data_spread)


def despread(composite, code, codelength):
    l = int(len(composite) / codelength)
    despread = applyCarrier(composite)
    # despread = despread*(code*-2.0+1)
    despread = despread * (code)
    recovered = []
    for i in range(l):
        recovered = np.append(recovered, 1.0 * sum(despread[i * codelength:i * codelength + codelength]) / codelength)
    recovered = np.repeat(recovered, codelength)
    # recovered = applyCarrier(recovered)
    return recovered


def despread2(composite, code, codelength):
    l = int(len(composite) / codelength)
    despread = applyCarrier2(composite)
    # despread = despread*(code*-2.0+1)
    despread = despread * (code)
    recovered = []
    for i in range(l):
        recovered = np.append(recovered, 1.0 * sum(despread[i * codelength:i * codelength + codelength]) / codelength)
    recovered = np.repeat(recovered, codelength)
    # recovered = applyCarrier(recovered)
    return recovered


def applyCarrier(data):
    N = len(data)  # Number of symbols to be sent.
    Fc = 100  # Carrier frequency.
    Fs = 1000  # Sampling frequency.
    tStep = 1 / Fs  # Width of each symbol (in sec).

    t = np.linspace(0, (N - 1) * tStep, N)
    carrier = np.cos(2 * np.pi * Fc * t)
    modData = np.multiply(data, carrier)
    return modData


def applyCarrier2(data):
    N = len(data)  # Number of symbols to be sent.
    Fc = 200  # Carrier frequency.
    Fs = 1000  # Sampling frequency.
    tStep = 1 / Fs  # Width of each symbol (in sec).

    t = np.linspace(0, (N - 1) * tStep, N)
    carrier = np.cos(2 * np.pi * Fc * t)
    modData = np.multiply(data, carrier)
    return modData


def applyCarrierModified(data, carrierFrequency):
    bitWiseCarrier = []
    N = len(data)  # Number of symbols to be sent.
    # N = 16  # Number of symbols to be sent.
    Fc = carrierFrequency  # Carrier frequency.
    Fs = 940  # Sampling frequency.
    tStep = 1 / Fs  # Width of each symbol (in sec).

    t = np.linspace(0, (N - 1) * tStep, N)
    carrier = 2 * np.cos(2 * np.pi * Fc * t)
    for i in range(len(data)):
        modData = np.multiply(data[i], carrier)
        bitWiseCarrier.append(modData)

    return bitWiseCarrier


def continuousFFT(signals):
    fftBin = {}
    for signal in signals:
        frequency, magnitude = applyfftModified(signal)
        magnitudes = fftBin.get(frequency)
        if magnitudes is None:
            magnitudes = [magnitude]
        else:
            magnitudes.append(magnitude)

        fftBin[frequency] = magnitudes
    return fftBin


def applyfft(signal):
    print('signal: ', signal)
    Fs = 1000
    N = len(signal)
    fStep = Fs / N
    tStep = 1 / Fs
    f0 = 100
    f = np.linspace(0, (N - 1) * fStep, N)
    t = np.linspace(0, (N - 1) * tStep, N)
    y = np.cos(2 * np.pi * f0 * t)
    X = np.fft.fft(signal)
    ValArray = []
    for v in X:
        ValArray.append(math.sqrt(v.real ** 2 + v.imag ** 2))
    print('FFT: ', ValArray)
    Xmag = np.abs(X) / N
    # Xmag = np.abs(X)
    print('FFT Mag: ', Xmag)
    fRealized = f[0: int(N / 2 + 1)]
    XmagRealized = 2 * Xmag[0: int(N / 2 + 1)]
    XmagRealized[0] = Xmag[0] / 2

    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
    ax1.plot(fRealized, XmagRealized, '.-')
    # ax2.plot(t, signal, '.-')
    plt.show()


def applyfftModified(signal):
    Fs = 1000
    N = len(signal)
    fStep = Fs / N
    tStep = 1 / Fs
    f = np.linspace(0, (N - 1) * fStep, N)
    # X = np.fft.fft(signal, N)
    X = np.fft.fft(signal)
    count = 0
    Xmag = np.abs(X) / N

    fRealized = f[0: int(N / 2 + 1)]
    XmagRealized = 2 * Xmag[0: int(N / 2 + 1)]
    XmagRealized[0] = Xmag[0] / 2
    magnitude = np.amax(XmagRealized)
    magnitudeIndex = np.where(XmagRealized == np.amax(XmagRealized))
    frequency = fRealized[magnitudeIndex[0][0]]

    # plt.plot(fRealized, XmagRealized, '.-')
    # plt.ylim([0, 50])
    # plt.show()

    return frequency, magnitude


def buildCompositeSignal(signals):
    compositeSignal = []
    for i in range(56):
        sum = 0
        for signal in signals:
            sum = sum + signal[i]
        compositeSignal.append(sum)
    return compositeSignal


# goldCodes = gold.getGoldCodesByPaperExample()
goldCodes = gold.getGoldCodesByWebExample()

g0 = np.where(goldCodes[0], 1, 0)
g1 = np.where(goldCodes[1], 1, 0)
g2 = np.where(goldCodes[2], 1, 0)
g3 = np.where(goldCodes[3], 1, 0)
# g30 = np.where(goldCodes[30], 1, 0)
g30 = np.where(goldCodes[4], 1, 0)
g5 = np.where(goldCodes[6], 1, 0)
g6 = np.where(goldCodes[6], 1, 0)
codelength = len(g0)  # 2^8 - 1 = 255
print('codelength: ', codelength)

# Primary user data
p, p_len = buildUserData(0x91, codelength)
# First secondary user and his code
q, q_len = buildUserData(0xc1, codelength)
q_code, q_spread = spreadUserData(q, q_len, g0)
# Second secondary user and her code
r, r_len = buildUserData(0xa5, codelength)
r_code, r_spread = spreadUserData(r, r_len, g30)
# Third secondary user and her code
s, s_len = buildUserData(0xb5, codelength)
s_code, s_spread = spreadUserData(s, s_len, g1)
# Fourth secondary user and her code
t, t_len = buildUserData(0xd5, codelength)
t_code, t_spread = spreadUserData(t, t_len, g2)
# Fifth secondary user and her code
u, u_len = buildUserData(0xe5, codelength)
u_code, u_spread = spreadUserData(u, u_len, g3)

carrier2UserData1, carrier2UserData1Len = buildUserData(0x101, codelength)
carrier2UserData1_code, carrier2UserData1_spread = spreadUserData2(carrier2UserData1, carrier2UserData1Len, g5)
carrier2UserData2, carrier2UserData2Len = buildUserData(0x120, codelength)
carrier2UserData2_code, carrier2UserData2_spread = spreadUserData2(carrier2UserData2, carrier2UserData2Len, g6)

# Composite sigal from all three users
# composite = (p*2-1) + (r_spread*2-1) + (q_spread*2-1) + (s_spread*2-1) + (t_spread*2-1) + + (u_spread*2-1)

# composite = p + r_spread + q_spread + s_spread + t_spread + u_spread
# composite = r_spread + q_spread + s_spread + t_spread + u_spread
# print('Composite: ', composite)
# composite2 = carrier2UserData1_spread + carrier2UserData2_spread

# if len(composite) < len(composite2):
#     c = composite2.copy()
#     c[:len(composite)] += composite
# else:
#     c = composite.copy()
#     c[:len(composite2)] += composite2
# applyfft(c)
# applyfft(composite)
# applyfftModified(composite)
#
# noise = np.random.normal(0, .1, composite.shape)

composite = buildCompositeSignal([r_spread, q_spread, s_spread, t_spread, u_spread])

value = continuousFFT(composite)
values = []
keys = []
for k, v in value.items():
    keys.append(k)
    values.append(v)

minValue = np.amin(values[0])
print(minValue)

normalizedValues = []

for v in values[0]:
    normalizedValues.append(math.ceil(v / minValue))

print(normalizedValues)
