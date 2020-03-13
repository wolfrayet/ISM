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
k230 = 0.005
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
def greybody(wavelength, T, N, b):
    freq = c / wavelength
    return N * blackbody(wavelength, T) * k230 * (freq/230e9)**b
def gray(wavelength, T, logN, b):
    freq = c / wavelength
    return logN + np.log(2*h/c**2) + (b+3)*np.log(freq) - np.log(np.exp(h*freq/k/T)-1) - b*np.log(230e9)
# select data
x = data[np.where((data[:,0]>2e5) & (data[:,0]<1e7))[0],0]      # wavelenth
y = data[np.where((data[:,0]>2e5) & (data[:,0]<1e7))[0],1]      # flux density
#y_error = np.ones(len(y))

y = y * 3.71e-11 / 0.870964   
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
popt, pcov = curve_fit(f, x, y, p0=[60.0,1.0e-14,1.5,2.0]) #, bounds=((10.0, 0, 1.25, 0.5),(170.0, 1e-5, 2.5, 5.5))
#popt2, pcov2 = curve_fit(greybody, x, y, p0=[20.0,1.0e-10,1]) #, bounds=((10.0,0,1.25,0),(170.0,np.inf,2.5,0.5))
popt3, pcov3 = curve_fit(gray, x, np.log(y), p0=[40.0, -40.0, 1.5],bounds=((10.0,-80,-1.0),(80.0,-20,2.5)))
#perr = np.sqrt(np.diag(pcov))
#perr2 = np.sqrt(np.diag(pcov2))
mse1 = np.sum((y-f(x, *popt))**2)/(len(y)-4)
#mse2 = np.sum((y-greybody(x, *popt2))**2)/(len(y)-3)
mse3 = np.sum((y-np.exp(gray(x, *popt3)))**2)/(len(y)-3)
chisq1 = chisquare(y, f(x, *popt), ddof=4)
#chisq2 = chisquare(y, greybody(x, *popt2), ddof=3)
chisq3 = chisquare(y, np.exp(gray(x, *popt3)), ddof=3)
print(popt, mse1, chisq1)
#print(popt2, mse2, chisq2)
print(popt3, mse3, chisq3)
fig = plt.figure()
x_plot = np.linspace(20,1200, num=150) * 1e-4
plt.plot(x*1e4, y, 'bo', label='data')
plt.plot(x_plot*1e4, f(x_plot, *popt), 'r-', label="Casey 2011, MSE={:.2e}".format(mse1))
#plt.plot(x*1e4, f(x, 60.0, 1e-16, 1.729, 1.20), 'y-', label='Casey 2011')
#plt.plot(x*1e4, greybody(x, *popt2), 'g-', label='greybody')
plt.plot(x_plot*1e4, np.exp(gray(x_plot, *popt3)), 'g-', label="Graybody, MSE={:.2e}".format(mse3))
plt.ylabel(r'$Flux\,(Jy)$', fontsize='large')
plt.xlabel(r'$Wavelength\,(\mu m)$', fontsize='large')
plt.xscale('log')
plt.title(fname.replace('_template_norm.sed',''))
plt.legend(loc='upper right', fontsize='medium', handletextpad=0.1)
plt.grid(True)
fig.savefig(fname.replace('_template_norm.sed','')+'_v1.png')
