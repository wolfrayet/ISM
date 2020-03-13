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

# fitting parameters
#a = float(2)
lambda_0 = float(200e-4)  # cgs

# fitting curve
def fbb(wavelength, Td, b):
    freq = c / wavelength
    exp1 = 1 - np.exp(- ((lambda_0/wavelength)**b))
    exp2 = np.exp(h*freq/k/Td) - 1
    return 2 * h * exp1 * freq**3 / exp2 / c**2
def f(wavelength, Td, N, b, a):
    lambda_c = 1 / ((26.6764 + 6.24629*a)**-2 + (1.90530e-04 + 7.24277e-05*a)*Td) * 1e-4
    power = ((wavelength/lambda_c)**a) * np.exp(- (wavelength/(lambda_c*3/4))**2)
    return N * (fbb(wavelength,Td, b) + fbb(lambda_c, Td, b)*power)
def blackbody(wavelength, T):
    freq = c / wavelength
    return 2 * h * freq**3 / c**2 / (np.exp(h*freq/k/T) - 1)
def greybody(wavelength, T, N, b, k230):
    freq = c / wavelength
    return N * blackbody(wavelength, T) * k230 * (freq/230e9)**b

# select data
x = data[np.where((data[:,0]>4e5) & (data[:,0]<1e7))[0],0]      # wavelenth
y = data[np.where((data[:,0]>4e5) & (data[:,0]<1e7))[0],1]      # flux density
#y_error = np.ones(len(y))

y = y * 3.71e-11 / 0.870964
# convert to real flux density
# M82 6250A = 1.73e-4 / 0.900673 erg/s/cm2/A 
# NGC 6090 6250A = 7.86e-11 / 0.923506 erg/s/cm2/A  
# NGC 6240 6250A = 3.66e-10 / 0.949810 erg/s/cm2/A 
# Arp 220 6250A = 1.62e-10 / 0.901856 erg/s/cm2/A 
# IRAS 22491 6250A = 1.14e-11 / 0.869906 erg/s/cm2/A  
# IRAS 20551 6250A = 3.71e-11 / 0.870964 erg/s/cm2/A    
x = x * 1e-8                    # convert to cgs
y = y * x**2 / c                # convert to erg/s/cm2/Hz
#y_error = y_error * 4.61e-6 * x**2 / c
# fitting
popt, pcov = curve_fit(f, x, y, p0=[30.0,1.0e-10,1.5,2.0]) #, bounds=((10.0, 0, 1.25, 0.5),(170.0, 1e-5, 2.5, 5.5))
popt2, pcov2 = curve_fit(greybody, x, y, p0=[30.0,1.0e-11,0.5,0.009]) #, bounds=((10.0,0,1.25,0),(170.0,np.inf,2.5,0.5))
perr = np.sqrt(np.diag(pcov))
perr2 = np.sqrt(np.diag(pcov2))
print(popt)
print(popt2)
mse1 = np.sum((y-f(x, *popt))**2)/(len(y)-4)
mse2 = np.sum((y-greybody(x, *popt2))**2)/(len(y)-4)
chisq1 = chisquare(y, f(x, *popt), ddof=4)
chisq2 = chisquare(y, greybody(x, *popt2), ddof=4)
print(mse1, chisq1)
print(mse2, chisq2)
fig = plt.figure()
plt.plot(x*1e4, y, 'bo', label='data')
plt.plot(x*1e4, f(x, *popt), 'r-', label='Casey 2011')
plt.plot(x*1e4, greybody(x, *popt2), 'g-', label='greybody')
#plt.plot(x*1e4, f(x, 60, 4.5e-10, 1.5, 2.5), 'y-')
plt.ylabel(r'$Flux\,(Jy)$', fontsize='large')
plt.xlabel(r'$Wavelength\,(\mu m)$', fontsize='large')
plt.xscale('log')
#plt.title("FIR fitting of "+fname.replace('.sed',''))
plt.legend(loc='upper right', fontsize='medium', handletextpad=0.1)
plt.grid(True)
fig.savefig(fname.replace('_template_norm.sed','')+'_v2.png')