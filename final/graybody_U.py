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
if len(sys.argv) < 2:
    print("Please inform the filename.")
    exit(1)
elif len(sys.argv) == 2:
    print("Please inform redshift and luminosity distance.")
    exit(1)
fname = sys.argv[1]
z = float(sys.argv[2])  # redshift
d = float(sys.argv[3])  # luminosity distance
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
k230 = 0.005
# fitting curve
def gray(wavelength, T, logN, b):
    freq = c / wavelength
    return logN + np.log(2*h/c**2) + (b+3)*np.log(freq) - np.log(np.exp(h*freq/k/T)-1) - b*np.log(230e9)
def Mpc2cm(mpc):
    return mpc * 3.086e+24
def mass(freq, flux, T, b, mpc):
    logk = np.log(k230) + b*np.log(freq/230e9)
    logM = -np.log(1e23) + np.log(flux) + 2*np.log(mpc) - logk - np.log(2*h*(freq**3)/(c**2)) - np.log(np.exp(h*freq/k/T)-1) - np.log(Msun) - np.log(1+z)
    return logM / np.log(10)

# select data
x = data[:,0]      # wavelenth: um
y = data[:,1]      # flux density: Jy
y_err = data[:,2]
# unit conversion
x = x * 1e-4 * (1+z)    # convert to cgs and redshift back

# fitting
popt, pcov = curve_fit(gray, x, np.log(y), p0=[20.0, 0.0, 1.5], sigma=y_err) #,bounds=((10.0,-40,-1.0),(80.0,40,2.5))
mse = np.sum((y-np.exp(gray(x, *popt)))**2)/(len(y)-3)
chisq = chisquare(y, np.exp(gray(x, *popt)), ddof=3)
perr = np.sqrt(np.diag(pcov))
print(popt, mse, chisq)
print(perr)

# calculation
M = mass(c/x[3], y[3], popt[0], popt[2], Mpc2cm(d))
print(M)
M = mass(c/x[3], np.exp(gray(x[3],*popt)), popt[0], popt[2], Mpc2cm(d))
print(M)
# plot
#x_plot = np.linspace(50,1200, num=100) * 1e-4
#fig = plt.figure()
#y1 = np.exp(gray(x_plot, *(popt + perr)))
#y2 = np.exp(gray(x_plot, *(popt - perr)))
#plt.errorbar(x*1e4, y, yerr=y_err, fmt='bo')
#plt.plot(x_plot*1e4, np.exp(gray(x_plot, *popt)), 'r-', label='graybody')
#plt.fill_between(x_plot*1e4, y2, y1, facecolor='grey', alpha=0.5)
#plt.ylabel(r'$Flux\,(Jy)$', fontsize='large')
#plt.xlabel(r'$Wavelength\,(\mu m)$', fontsize='large')
#plt.xscale('log')
#plt.xlim([50,1000])
#plt.legend(loc='upper right', fontsize='medium', handletextpad=0.1)
#plt.grid(True)
#plt.title(fname.replace('.sed',''))
#fig.savefig(fname.replace('.sed','')+'_v1.png')
