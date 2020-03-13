import numpy as np
import astropy.constants as const
import matplotlib.pyplot as plt
from scipy.integrate import quad
# constants
c = const.c.value * 100     # cm/s
h = const.h.value * 1e7     # Planck: erg*s
k = const.k_B.value * 1e7   # Boltzmann: erg/T
pc = const.pc.value * 100
d = 1000
m = 1.00794 * const.u.value * 1000      # H mass: g
f0 = 1420.41e6              # H-21 freq: Hz
a10 = 2.86889e-15           # spontaneous emission: Hz 
Tk = 50                     # kinectic temperature: K
# functions 
def stimulate(a10):
    return a10 * c**2 / 2 / h / f0**3
def absorption(b10, ratio):
    return b10 * ratio
def Voigit(y, a, u):
    return np.exp(-y**2) / (a**2 + (u-y)**2)
def LineProfile(freq, T):
    width = np.sqrt(2*k*T/m) * f0 / c
    #A = a10 / 4 / np.pi / width
    #u = (freq-f0) / width
    #profile = quad(lambda x: Voigit(x, A, u), 0, np.inf)
    #print(profile)
    #result = profile[1] * A / np.pi * 2
    #return result / width / np.sqrt(np.pi)
    return np.exp(-(freq-f0)**2/width**2) / width / np.sqrt(np.pi)
def BlackBody(freq, T):
    return 2 * h * freq**3 / c**2 / (np.exp(h*freq/k/T) - 1)
def alpha(freq, T, b, n):
    a0 = h * freq * n * b / 4 / np.pi
    return a0 * (1 - np.exp(-h*f0/k/T)) #* LineProfile(freq, T)
def emission(freq, T, a, n):
    j0 = h * freq * n * a / 4 / np.pi
    return j0 #* LineProfile(freq, T)
def intensity(freq, T, Tb, d, a):
    tau = a * d * pc
    if Tb == 0:
        I = 0
    else:
        I = BlackBody(freq, Tb) * np.exp(-tau)
    S = BlackBody(freq, T) * (1 - np.exp(-tau))
    return (I + S)
# calculation
b10 = stimulate(a10)
b01 = absorption(b10, 3)
print('a10, b10, b01:', a10, b10, b01)
kappa = alpha(f0, Tk, b01, 0.25)
j = emission(f0, Tk, a10, 0.75)
print('absorption:', kappa)
print('emission:', j)
print('no background:', intensity(f0, Tk, 0, d, kappa))
print('2.7K:', intensity(f0, Tk, 2.7, d, kappa))
# plot
x = np.logspace(-5, 3, num=1000)
fig = plt.figure()
plt.plot(x, intensity(f0, Tk, 0, x, kappa), color='#1f77b4', label='no background')
plt.plot(x, intensity(f0, Tk, 2.7, x, kappa), color='#ff7f0e', label='2.7K')
plt.xscale('log')
plt.yscale('log')
plt.grid(b=True, which='both', axis='both')
plt.legend(loc=2)
plt.title('H-21cm emission intensity vs. cloud thickness, T=50K')
plt.xlabel('cloud thickness (pc)')
plt.ylabel('intensity (erg / s / ster / Hz)')
fig.savefig('H21_delta_profile.png')
