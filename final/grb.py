import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.constants import codata
from scipy import integrate
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preview'] = True
if len(sys.argv) < 2:
    print("Please inform the filename.")
    exit(1)
fname = sys.argv[1]
try:
    data = np.loadtxt(fname, dtype='float')
except IOError:
    print("File '%s' does not exit.",fname)
    exit(1)
# scientific constants
c = codata.value('speed of light in vacuum') * 100  # cgs
h = codata.value('Planck constant') * 1e7           # cgs
k = codata.value('Boltzmann constant') * 1e7        # cgs
Msun = 1.9885e33                                    # solar mass cgs
l_sun = 3.828e+33
k230 = 0.005
a = float(2)
lambda_0 = float(200e-4)    # cgs
N = np.ones(3)
# fitting curve
def fbb(freq, Td, b):
    wavelength = c / freq
    exp1 = 1 - np.exp(- ((lambda_0/wavelength)**b))
    exp2 = np.exp(h*freq/k/Td) - 1
    return 2 * h * exp1 * freq**3 / exp2 / c**2
def f(freq, Td, b):
    wavelength = c / freq
    lambda_c = 1 / ((26.6764 + 6.24629*a)**-2 + (1.90530e-04 + 7.24277e-05*a)*Td) * 1e-4
    freq_c = c / lambda_c
    power = ((wavelength/lambda_c)**a) * np.exp(- (wavelength/(lambda_c*3/4))**2)
    return fbb(freq,Td, b) + fbb(freq_c, Td, b)*power
def mass(freq, flux, Td, b, mpc):
    logk = np.log(k230) + b*np.log(freq/230e9)
    logM = -np.log(1e23) + np.log(flux) + 2*np.log(mpc) - logk - np.log(2*h*(freq**3)/(c**2)) - np.log(np.exp(h*freq/k/Td)-1) - np.log(Msun) - np.log(1+z)
    return logM / np.log(10)
# data
z = data[:,1]   # redshift
d = data[:,2]*3.086e+24   # luminosity distance: Mpc  
y = data[:,3]*1e-6  # Jy
y_err = data[:,4]*1e-6
t = np.array([29.68, 43.37, 34.42, 34.98, 34.69, 39.97, 42.67, 35.63, 33.06])
t_err = np.array([1.55, 1.21, 0.64, 0.49, 0.88, 0.69, 3.03, 0.63, 0.22])
b_est = np.array([1.27, 1.39, 1.33, 1.64, 1.33, 1.32, 1.32, 1.44, 1.79])
b_err = np.array([0.13, 0.04, 0.04, 0.03, 0.05, 0.03, 0.11, 0.04, 0.02])
up = c / 8.0e-4
low = c / 1000.0e-4
# calculation
T0 = np.average(t, weights=(1/t_err))
T0_err = np.sum(t_err)/np.sqrt(len(t_err))
b0 = np.average(b_est, weights=(1/b_err))
#b_err = np.sum(b_err)/np.sqrt(len(b_err))
x = 343.5e9 * (1+z)
N = y / f(x, T0, b0)
IR = N * integrate.quad(f, low, up, args=(T0, b0))[0] * 4 * np.pi * d**2 * 1e-23 / (l_sun)
IR_up = N * integrate.quad(f, low, up, args=(T0+T0_err, b0))[0] * 4 * np.pi * d**2 * 1e-23 / (l_sun)
IR_do = N * integrate.quad(f, low, up, args=(T0-T0_err, b0))[0] * 4 * np.pi * d**2 * 1e-23 / (l_sun)
sfr = IR * 1.7e-10
sfr_up = IR_up * 1.7e-10
sfr_do = IR_do * 1.7e-10
M = mass(x, y, T0, b0, d)
print(b0)
print(T0, T0_err)
print(IR)
print(IR_up-IR)
print(IR-IR_do)
print(sfr)
print(sfr_up-sfr)
print(sfr-sfr_do)
print(M)
yerr0 = y_err[0]
pt = np.arange(4., 1200., 1) * 1e-4
plt.plot(c/x[0]*1e4, y[0], 'bo', label='data')
plt.errorbar(c/x[0]*1e4, y[0], yerr=yerr0, fmt='bo')
plt.plot(pt*1e4, N[0]*f(c/pt, T0, b0), 'r-', label='Casey')
plt.fill_between(pt*1e4, N[0]*f(c/pt, T0-T0_err, b0), N[0]*f(c/pt, T0+T0_err, b0), color='grey', alpha=0.2)
plt.ylabel(r'$Flux\,(Jy)$', fontsize='large')
plt.xlabel(r'$Wavelength\,(\mu m)$', fontsize='large')
plt.yscale('log')
plt.xscale('log')
plt.title('IR SED of GRB 080607')
plt.legend(loc='upper right', fontsize='medium', handletextpad=0.1)
plt.grid(True)
plt.show()