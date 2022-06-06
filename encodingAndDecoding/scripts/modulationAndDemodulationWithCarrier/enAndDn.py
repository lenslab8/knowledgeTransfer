import pylab as plt
import numpy as np

from numpy import fft

from scripts.mlsMatLabLike.Gold import Gold
from scripts.mlsPaper.gold import getGoldCodesByPaperExample


def cconv(x, y):
    """Calculate the circular convolution of 1-D input numpy arrays using DFT
    From the Signal Processing Library: http://mubeta06.github.io/python/sp/filter.html
    """
    return fft.ifft(fft.fft(x)*fft.fft(y))

def ccorr(x, y):
    """Calculate the circular correlation of 1-D input numpy arrays using DFT
    From the Signal Processing Library: http://mubeta06.github.io/python/sp/filter.html
    """
    return fft.ifft(fft.fft(x)*fft.fft(y).conj())

def despread(composite, code, codelength):
    l = int(len(composite)/codelength)
    despread = composite*(code*-2.0+1)
    recovered = []
    for i in range(l):
        recovered = np.append(recovered, 1.0*sum(despread[i*codelength:i*codelength+codelength])/codelength)
    recovered = np.repeat(recovered, codelength)
    return recovered


def bitfield(n):
    """Convert integer into bitfield (as list)
    From StackOverflow: http://stackoverflow.com/a/10322018/
    """
    return [int(digit) for digit in bin(n)[2:]]

def buildUserData(data, goldCode):
    data = data
    data = np.array(bitfield(data))
    data_len = len(data)
    data = np.repeat(data, codelength)
    data_code = []
    for i in range(data_len):
        data_code = np.append(data_code, goldCode)

    data_spread = np.logical_xor(data_code, data).astype(int)
    return (data, data_code, data_spread)

def carrierTest(data):
    print('Data ', data)
    N = len(data)  # Number of symbols to be sent.
    Fc = 100  # Carrier frequency.
    Fs = 1000  # Sampling frequency.
    tStep = 1 / Fs  # Width of each symbol (in sec).

    t = np.linspace(0, (N-1)*tStep, N)
    carrier =  np.cos(2 * np.pi * Fc * t)
    modData = np.multiply(data, carrier)
    dModData = np.multiply(modData, carrier)
    print("DmodData ", dModData)
    applyfft(modData)

def applyfft(signal):
    Fs = 1000
    N = len(signal)
    fStep = Fs / N
    tStep = 1 / Fs
    f0 = 100
    f = np.linspace(0, (N-1) * fStep, N)
    print('f', f)
    t = np.linspace(0, (N-1) * tStep, N)
    y = np.cos(2 * np.pi * f0 * t)
    X = np.fft.fft(signal)
    Xmag = np.abs(X) / N
    fRealized = f[0: int(N/2 + 1)]
    XmagRealized = 2 * Xmag[0: int(N/2 + 1)]
    XmagRealized[0] = Xmag[0] / 2

    fig, [ax1, ax2] = plt.subplots(nrows = 2, ncols = 1)
    ax1.plot(fRealized, XmagRealized, '.-')
    #ax2.plot(t, signal, '.-')
    plt.show()



goldCodes = getGoldCodesByPaperExample()

g0 = np.where(goldCodes[0], 1, 0)
g30 = np.where(goldCodes[30], 1, 0)


# Two Gold codes. See
# Gold, R. "Optimal binary sequences for spread spectrum multiplexing (Corresp.)"
# IEEE Transactions on Information Theory. (October 1967)
# g0 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1,
#        0, 0, 1, 0, 1, 1, 0, 0], dtype=int)
# g30 = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0,
#        0, 1, 1, 1, 1, 1, 0, 1])

codelength = len(g0) # 2^8 - 1 = 255

# Primary user data
p = 0x91
p = np.array(bitfield(p))
p_len = len(p)

p = np.repeat(p, codelength)

# First secondary user and his code
q = 0xc1
q = np.array(bitfield(q))
q_len = len(q)
q = np.repeat(q, codelength)
q_code = []
for i in range(q_len):
  q_code = np.append(q_code, g30)

q_spread = np.logical_xor(q_code, q).astype(int)

# Second secondary user and her code
r = 0xa5
r = np.array(bitfield(r))
r_len = len(r)
r = np.repeat(r, codelength)
r_code = []
for i in range(r_len):
  r_code = np.append(r_code, g0)

r_spread = np.logical_xor(r_code, r).astype(int)

carrierTest(r_spread)

# Composite sigal from all three users
composite = (p*2-1) + (r_spread*2-1) + (q_spread*2-1)

p_recovered = np.array([], dtype = float)
for i in range(p_len):
    p_recovered = np.append(p_recovered, 1.0*sum(composite[i*codelength:i*codelength+codelength])/codelength)
p_recovered = np.repeat(p_recovered, codelength)

r_recovered = despread(composite, r_code, codelength)
#q_recovered = despread(composite, q_code, codelength)

# plt.figure()
# plt.subplot(3,2,1)
# plt.title('Autocorrelation g0')
# plt.plot((np.roll(ccorr(g0, g0).real, int(len(g0)/2-1))), color="green")
# plt.xlim(0, len(g0))
# plt.ylim(0, 22)
# plt.subplot(3,2,2)
# plt.title('Autocorrelation g30')
# plt.plot((np.roll(ccorr(g30, g30).real, int(len(g30)/2-1))), color="purple")
# plt.xlim(0, len(g30))
# plt.ylim(0, 22)
# plt.subplot(3,2,3)
# plt.title('Crosscorrelation g0 g30')
# plt.plot((np.roll(ccorr(g0, g30).real, int(len(g0)/2-1))))
# plt.xlim(0, len(g0))
# plt.ylim(0, 22)
# plt.subplot(3,2,4)
# plt.title('Crosscorrelation g30 g0')
# plt.plot((np.roll(ccorr(g30, g0).real, int(len(g30)/2-1))))
# plt.xlim(0, len(g30))
# plt.ylim(0, 22)
# plt.subplot(3,2,5)
# plt.title('g0')
# plt.step(range(len(g0)), g0, color="green")
# plt.xlim(0, len(g0))
# plt.ylim(-0.5, 1.5)
# plt.subplot(3,2,6)
# plt.title('g30')
# plt.step(range(len(g30)), g30, color="purple")
# plt.xlim(0, len(g30))
# plt.ylim(-0.5, 1.5)
# plt.subplots_adjust(hspace=.5)
# plt.show()
#
#
#
# f, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, sharex=True, sharey=True)
# ax0.set_title('Signals and codes of 3 users')
# ax0.step(range(len(p)),p*2-1)
# ax0.axis((0,len(r),-1.5,1.5))
# ax1.step(range(len(r)),r*2-1, color="green")
# ax2.step(range(len(r_code)),r_code*2-1, color="green")
# ax3.step(range(len(q)),q*2-1, color="purple")
# ax4.step(range(len(q_code)),q_code*2-1, color="purple")
# f.subplots_adjust(hspace=0.1)
# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
# plt.show()
#
# f, (ax0, ax1, ax2) = plt.subplots(3, sharex=True, sharey=True)
# ax0.set_title('Composite signal, and composite multiplied by code')
# ax0.step(range(len(composite)),composite, color="brown")
# ax0.axis((0,len(r),-4.5,4.5))
# ax1.step(range(len(composite)),composite*r_code, color="green")
# ax2.step(range(len(composite)),composite*q_code, color="purple")
# f.subplots_adjust(hspace=0.1)
# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
# plt.show()
#
#
# g, (ax0, ax1, ax2) = plt.subplots(3, sharex=True, sharey=True)
# ax0.set_title('Recovered signal of 3 users')
# ax0.step(range(len(p)),p*2-1, color='gray')
# ax0.step(range(len(p_recovered)),p_recovered)
# ax0.axis((0,len(r),-2.5,2.5))
# ax0.axhline(color="gray", linestyle="dashed")
# ax1.step(range(len(r)),r*2-1, color='gray')
# ax1.step(range(len(r_recovered)),r_recovered, color="green")
# ax1.axhline(color="gray", linestyle="dashed")
# ax2.step(range(len(q)),q*2-1, color='gray')
# ax2.step(range(len(q_recovered)),q_recovered, color="purple")
# ax2.axhline(color="gray", linestyle="dashed")
# g.subplots_adjust(hspace=0.1)
# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
# plt.show()

