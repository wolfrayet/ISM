import numpy as np
import matplotlib.pyplot as plt
# --------------- files ------------------------------------------
#file_ngc = '2MASS.txt'
file_ngc = 'ds9.tsv'
file_mainseq = 'MainSeqJHK.txt'
file_giant = 'GiantJHK.txt'
file_super = 'SuperGiantJHK.txt'
file_Cext = 'Cext.txt'
lambda_band = [1.235, 1.662, 2.159]     # wavelength of J H K band
# --------------- y=ax+b -----------------------------------------
def f(x, a, x0, y0):
    b = y0 - a*x0
    return a*x + b
# --------------- load file --------------------------------------
#data = np.loadtxt(file_ngc, skiprows=3, usecols=[0,1,2,3,4])
data = np.loadtxt(file_ngc, delimiter='\t',skiprows=1, usecols=[0,1,3,5,7])
mainseq = np.loadtxt(file_mainseq, skiprows=1)
giant = np.loadtxt(file_giant, skiprows=1)
superg = np.loadtxt(file_super, skiprows=1)
c_ext_model = np.flip(np.loadtxt(file_Cext, skiprows=3, usecols=[0,3]), axis=0)
# --------------- main -------------------------------------------
j_h = data[:,2] - data[:,3]
h_k = data[:,3] - data[:,4]
result = np.vstack((j_h,h_k)).T
data = np.concatenate((data, result), axis=1)
stand = np.concatenate((mainseq, giant, superg), axis=0)
c_ext = np.interp(lambda_band, c_ext_model[:,0], c_ext_model[:,1])
slope_ext = (c_ext[0]-c_ext[1])/(c_ext[1]-c_ext[2])
JHmin = np.argmin(stand[:,0], axis=0)
JHmax = np.argmax(stand[:,0], axis=0)
YSO = j_h < f(h_k, slope_ext, stand[JHmin,1], stand[JHmin,0])
# --------------- output -----------------------------------------
data = data[:,[0,1,5,6]]
colormax = np.linspace(stand[JHmax,1], 2, 100)
colormin = np.linspace(stand[JHmin,1], 2, 100)
np.savetxt('YSO.txt', data[YSO,:], delimiter=' ', header='ra dec J-H H-K', fmt='%.4f')
JH = (c_ext[0]-c_ext[1])*10/3.1/(6.415e-22-4.896e-22)
HK = (c_ext[1]-c_ext[2])*10/3.1/(6.415e-22-4.896e-22)
plt.scatter(data[:,3], data[:,2], marker='+', label='sources')
plt.scatter(data[YSO,3], data[YSO,2], marker='+', label='YSO candidates')
plt.scatter(mainseq[:,1], mainseq[:,0], marker='.', label='intrinsic stars')
plt.scatter(giant[:,1], giant[:,0], marker='.', label='intrinsic giants')
plt.scatter(superg[:,1], superg[:,0], marker='.', label='instrinsic supergiants')
plt.arrow(-0.5, 1.5, HK, JH,width=0.025, color='k', label='Av=10, mag=(0.67, 1.14)')
plt.text(-1.3,2.5, 'Av=10, mag=(0.67,1.14)')
plt.plot(colormax, f(colormax, slope_ext, stand[JHmax,1], stand[JHmax,0]), '--', c='k')
plt.plot(colormin, f(colormin, slope_ext, stand[JHmin,1], stand[JHmin,0]), '--', c='k')
plt.title('NGC1333')
plt.xlabel('H-K (mag.)')
plt.ylabel('J-H (mag.)')
plt.legend(loc='upper left')
plt.show()
