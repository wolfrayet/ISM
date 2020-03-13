import numpy as np
import astropy.io.fits as fits
import astropy.constants as const
import matplotlib.pyplot as plt
# import fits
file_3 = 'ALMA_Band3_regrid_to_Band6.fits'
file_63 = 'ALMA_Band6_smoothed_to_Band3.fits'
file_67 = 'ALMA_Band6_regrid_to_Band7.fits'
file_7 = 'ALMA_Band7_smoothed_to_Band6.fits'
b3 = fits.getdata(file_3)
b3_hd = fits.getheader(file_3)
b63 = fits.getdata(file_63)
b63_hd = fits.getheader(file_63)
b67 = fits.getdata(file_67)
b67_hd = fits.getheader(file_67)
b7 = fits.getdata(file_7)
b7_hd = fits.getheader(file_7)
# parameters
f3 = 101.9e9        # Hz
f6 = 233.0e9
f7 = 343.5e9
f230 = 230e9
Td = 25             # K
kappa230 = 0.009    # cm^2 / g
mu =2.8
m = 1.00794 * const.u.value * 1000      # g
m_sun = const.M_sun.value * 1000        # g
c = const.c.value * 100                 # cm / s
h = const.h.value * 1e7                 # erg s
k = const.k_B.value * 1e7               # erg / K
d = 140 * const.pc.value * 100     # cm
# functions
def deg2rad(deg):
    return deg * 0.0174533
def std_mask(image):
    noise = np.std(image, dtype=np.float64, ddof=1)
    image[image<noise] = 6e-06
def beam_size(header):
    major = deg2rad(float(header['BMAJ']))
    minor = deg2rad(float(header['BMIN']))
    return np.pi * major * minor / 4 / np.log(2)
def beta_pair(s1, s2, freq1, freq2):
    alpha = np.divide(np.log(np.divide(s1, s2)), np.log(freq1/freq2))
    beta = alpha - 2
    beta[s1<=6e-05] = -100
    beta[s2<=6e-05] = -100
    print('done')
    return beta
def kappa_freq(freq, beta):
    return kappa230 * (freq/f230)**beta
def blackbody(freq, T):
    return 2 * h * freq**3 / c**2 / (np.exp(h*freq/k/T) - 1)
def column_density(image, freq, beta, header, T, pix_size):
    beam = beam_size(header)
    pix_per_beam = (deg2rad(pix_size/3600))**2 / beam
    kappa = kappa_freq(freq, beta)
    kappa[beta<-3] = 1
    num = 1 / blackbody(freq, T) / mu / m / beam
    density = np.divide(image * num * 1e-23, kappa) #* pix_per_beam
    density[beta<-3] = 0
    print(1/pix_per_beam)
    return density
def total_mass(density, pix_size):
    pix_area = deg2rad(pix_size/3600) ** 2
    area = d**2 * pix_area
    mass = np.sum(density)
    return  mu * m * mass * area / m_sun
# calculation
std_mask(b7)
std_mask(b67)
std_mask(b63)
std_mask(b3)
beta_76 = beta_pair(b7, b67, f7, f6)
beta_36 = beta_pair(b3, b63, f3, f6)
density_7 = column_density(b7, f7, beta_76, b7_hd, Td, 0.0035)
density_67 = column_density(b67, f6, beta_76, b67_hd, Td, 0.0035)
density_63 = column_density(b63, f6, beta_36, b63_hd, Td, 0.005)
density_3 = column_density(b3, f3, beta_36, b3_hd, Td, 0.005)
mass_7 = total_mass(density_7, 0.0035)
print(mass_7)
mass_67 = total_mass(density_67, 0.0035)
print(mass_67)
mass_63 = total_mass(density_63, 0.005)
print(mass_63)
mass_3 = total_mass(density_3, 0.005)
print(mass_3)
# plot map
fig = plt.figure()
#plt.imshow(beta_36, cmap='jet', vmin=-0.5)
plt.imshow(density_3, cmap='jet', vmin=0)
plt.colorbar()
plt.title('Band3 Column Density')
#plt.show()
#flat36 = beta_36.flatten()
#plt.hist(flat36, range=(-2,5), bins=50)
#plt.title('Band6 & Band3')
fig.savefig('hist_b63.png')

