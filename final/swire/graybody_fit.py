import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.constants import codata
from scipy.optimize import curve_fit
from scipy.stats import chisquare
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preview'] = True
# read file from command line (use np.loadtxt, output ndarray)
if len(sys.argv) != 2:
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

# fitting curve
def gray(wavelength, T, logN, b):
    freq = c / wavelength
    return logN + np.log(2*h/c**2) + (b+3)*np.log(freq) - np.log(np.exp(h*freq/k/T)-1) - b*np.log(230e9)
# select data
x = data[np.where((data[:,0]>6e5) & (data[:,0]<1e7))[0],0]      # wavelenth
y = data[np.where((data[:,0]>6e5) & (data[:,0]<1e7))[0],1]      # flux density
#y_error = np.ones(len(y))

y = y * 3.66e-10 / 0.949810     
# convert to real flux density
# M82 6250A = 1.73e-4 / 0.900673 erg/s/cm2/A 
# NGC 6090 6250A = 7.86e-11 / 0.923506 erg/s/cm2/A  
# NGC 6240 6250A = 3.66e-10 / 0.949810 erg/s/cm2/A  
# IRAS 22491 6250A = 1.14e-11 / 0.869906 erg/s/cm2/A  
# IRAS 20551 6250A = 3.71e-11 / 0.870964 erg/s/cm2/A  
# Arp 220 6250A = 1.62e-10 / 0.901856 erg/s/cm2/A  
x = x * 1e-8                    # convert to cgs
y = y * x**2 / c                # convert to erg/s/cm2/Hz
#y_error = y_error * 4.61e-6 * x**2 / c
# fitting
popt3, pcov3 = curve_fit(gray, x, np.log(y), p0=[10.0, -28.0, 1.5],bounds=((10.0,-40,-1.0),(80.0,-20,2.5)))
mse3 = np.sum((y-np.exp(gray(x, *popt3)))**2)/(len(y)-3)
chisq3 = chisquare(y, np.exp(gray(x, *popt3)), ddof=3)
print(popt3, mse3, chisq3)
fig = plt.figure()
plt.plot(x*1e4, y, 'bo', label='data')
plt.plot(x*1e4, np.exp(gray(x, *popt3)), 'r-', label='graybody')
plt.ylabel(r'$Flux\,(Jy)$', fontsize='large')
plt.xlabel(r'$Wavelength\,(\mu m)$', fontsize='large')
plt.xscale('log')
plt.title(fname.replace('_template_norm.sed',''))
plt.legend(loc='upper right', fontsize='medium', handletextpad=0.1)
plt.grid(True)
fig.savefig(fname.replace('_template_norm.sed','')+'_v1.png')
