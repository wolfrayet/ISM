import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import codata
c = codata.value('speed of light in vacuum') 
def f(x, k, b):
    return k + b*x
x = np.array([5437.0, 6240.0, 7600.0, 15280.0]) * 1e-10
x = c / x
y = np.array([0.08, 0.08, 0.12, 0.48]) * 1e-6
y_err = np.array([0.02, 0.04, 0.03, 0.01]) * 1e-6
log_yerr_up = np.log(y+y_err) - np.log(y)
log_yerr_do = np.log(y) - np.log(y-y_err) 
popt, pcov = curve_fit(f, np.log(x), np.log(y), p0=(2.0, -1.0), sigma=y_err)
print(popt)
perr = np.sqrt(np.diag(pcov))
print(perr)
beta = popt[1]
print(1.97*(beta+2.3))
print(1.97*perr[1])
x_test = np.linspace(-2.5, 2.0, 50)
y1 = 1.71*(10**(0.4*1.97*(x_test+2.3))-1)
y2 = 1.71*(10**(0.4*0.91*(x_test+2.3))-1)
fig = plt.figure()
plt.plot(np.log(x), np.log(y), 'bo', label='data')
plt.errorbar(np.log(x), np.log(y), yerr=[log_yerr_do,log_yerr_up], fmt='bo')
plt.plot(np.log(x), f(np.log(x), *popt), 'r-',label='regression')
#plt.plot(x_test, y1, 'k-', label='Caltzetti')
#plt.plot(x_test, y2, 'r-', label='SMC')
#plt.errorbar(beta, 35.05/19.0, xerr=perr[1], yerr=[[11.45/19.00], [15.12/19.00]], fmt='-o')
#plt.yscale('log')
#plt.xlabel('beta')
#plt.ylabel('log(IRX)')
#plt.title('IRX-beta extinction law')
plt.title('UV observation of GRB 080607, beta=%.2f'%(popt[1]))
plt.xlabel('log(freq) (Hz)')
plt.ylabel('log(flux)')
plt.legend(loc='upper right', fontsize='medium', handletextpad=0.1)
plt.show()