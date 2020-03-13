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
if len(sys.argv) <= 2:
    print("Please inform the filename.")
    exit(1)
fname = sys.argv[1]
z = float(sys.argv[2])
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
a = float(2)
k230 = 0.005
lambda_0 = float(200e-4)  # cgs

# fitting curve
def fbb(wavelength, Td, b):
    freq = c / wavelength
    exp1 = 1 - np.exp(- ((lambda_0/wavelength)**b))
    exp2 = np.exp(h*freq/k/Td) - 1
    return 2 * h * exp1 * freq**3 / exp2 / c**2
def f(wavelength, Td, N, b):
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
x = data[:,0]      # wavelenth
y = data[:,1]*1e-6      # flux density
y_err = data[:,2]*1e-6
x = x * 1e-4 / (1+z)                    # convert to cgs

# fitting
#popt, pcov = curve_fit(f, x, y, p0=[20.0,1e4,1.5], sigma=y_err) #, bounds=((10.0, 0, 1.25, 0.5),(170.0, 1e-5, 2.5, 5.5))
popt3, pcov3 = curve_fit(gray, x, np.log(y), p0=[30.0, 0.0, 1.5], sigma=y_err) #, sigma=y_err
#perr = np.sqrt(np.diag(pcov))
perr3 = np.sqrt(np.diag(pcov3))
#mse1 = np.sum((y-f(x, *popt))**2)/(len(y)-4)
mse3 = np.sum((y-np.exp(gray(x, *popt3)))**2)/(len(y)-3)
#chisq1 = chisquare(y, f(x, *popt), ddof=4)
chisq3 = chisquare(y, np.exp(gray(x, *popt3)), ddof=3)
#print(popt, perr, mse1, chisq1)
print(popt3, perr3, mse3, chisq3)
fig = plt.figure()
x_plot = np.linspace(50,1200, num=150) * 1e-4
plt.errorbar(x*1e4, y, yerr=y_err, fmt='bo')
#plt.plot(x_plot*1e4, f(x_plot, *popt), 'r-', label='Casey 2011, T=%.2e, b=%.2e, MSE=%.2e' %(popt[0],popt[2],mse1))
plt.plot(x_plot*1e4, np.exp(gray(x_plot, *popt3)), 'r-', label='Casey, T=%.2e, b=%.2e, MSE=%.2e' %(popt3[0],popt3[2],mse3))
plt.ylabel(r'$Flux\,(Jy)$', fontsize='large')
plt.xlabel(r'$Wavelength\,(\mu m)$', fontsize='large')
plt.xscale('log')
plt.title(fname.replace('.sed',''))
plt.legend(loc='upper right', fontsize='medium', handletextpad=0.1)
plt.grid(True)
fig.savefig(fname.replace('.sed','')+'_v2.png')
