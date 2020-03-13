import numpy as np
#import astropy.io.fits as pyfits
#from astropy.wcs import WCS
#import matplotlib.pyplot as plt
from scipy.constants import codata
# read images and header files
#b3 = pyfits.getdata('ALMA_Band3_regrid_to_Band6.fits')
#b3_hd = pyfits.getheader('ALMA_Band3_regrid_to_Band6.fits')

#b6 = pyfits.getdata('ALMA_Band6_regrid_to_Band7.fits')
#b6_hd = pyfits.getheader('ALMA_Band6_regrid_to_Band7.fits')

#b6_to_3 = pyfits.getdata('ALMA_Band6_smoothed_to_Band3.fits')
#b6_to_3_hd = pyfits.getheader('ALMA_Band6_smoothed_to_Band3.fits')

#b7 = pyfits.getdata('ALMA_Band7_smoothed_to_Band6.fits') 
#b7_hd = pyfits.getheader('ALMA_Band7_smoothed_to_Band6.fits')

#wcs_76 = WCS(b7_hd)
#wcs_36 = WCS(b6_to_3_hd)
# frequency of each bands (GHz)
f3 = 90.0e9
f6 = 233.0e9
f7 = 343.5e9
# some constants
kappa230 = 0.005        # cm2/g
nu = 2.33                # mean molecular weight per H2
m = 1.674e-24           # mass of hydrogrn atom in cgs 
Td = 25.0               # dust temperature in K
c = codata.value('speed of light in vacuum') * 100  # cgs
h = codata.value('Planck constant') * 1e7       # cgs
k = codata.value('Boltzmann constant') * 1e7    # cgs 
m_sol = 1.989e33        # solar mass in cgs
d = 450 * c * 31536000 / 3.086e+20  # over 100pc
print(k/h * 1e-09 * 25)
def beam_size(header):
    major = header['BMAJ'] / 3600.0
    minor = header['BMIN'] / 3600.0
    return major * minor / 100
# apply Rayleigh-Jeans Approx.
def beta_pair(image1, image2, freq1, freq2):
    alpha = np.divide(np.log10(np.divide(image1, image2)), np.log10(freq1/freq2))
    beta = alpha - 2.0
    beta[image1<5e-05] = -100
    beta[image2<5e-05] = -100
    return beta

# column density of each block
def column_density(image, freq, beta, header):
    beam = beam_size(header)
    kappa = 100 * kappa230 * (freq/230.0e9) ** beta
    kappa[beta < -3] = 1
    wave = c / freq * 10 # unit: mm
    planck = 2.02e23 * (np.exp(1.439/2.5/wave)-1) * (wave**3) / beam
    density = planck * np.divide(image, kappa) 
    density[beta < -3] = 0
    return density
#b6[b6<0.0] = 1e-09
#b7[b7<0.0] = 1e-09
#b3[b3<0.0] = 1e-09
#b6_to_3[b6_to_3<0.0] = 1e-09

#beta_76 = beta_pair(b7, b6, f7, f6)
#beta_36 = beta_pair(b3, b6_to_3, f3, f6)
#density_7 = column_density(b7, f7, beta_76, b7_hd)
#density_6 = column_density(b6, f6, beta_76, b6_hd)
#density_3 = column_density(b3, f3, beta_36, b3_hd)
#mass_7 = 0.12 * 2.8792928e-16 * np.sum(np.divide(density_7, 2.02e23)) * d**2
#mass_6 = 0.12 * 2.8792928e-16 * np.sum(np.divide(density_6, 2.02e23)) * d**2
#mass_3 = 0.12 * 5.8761057e-16 * np.sum(np.divide(density_3, 2.02e23)) * d**2
#print(mass_7)
#print(mass_6)
#print(mass_3)
#pyfits.writeto('beta_76.fits', beta_76, b6_hd, overwrite=True)
#pyfits.writeto('beta_36.fits', beta_36, b6_to_3_hd, overwrite=True)

# beta map -------------------------------------------------------
#plt.subplot(projection=wcs_76)
#plt.imshow(beta_76, cmap='jet', clim=(-3,4.5))
#plt.subplot(projection=wcs_36)
#plt.imshow(beta_36, cmap='jet', vmin=-0.5)
# ---------------------------------------------------------------
#plt.imshow(density_7, cmap='jet', vmin=0)
#plt.imshow(density_6, cmap='jet', vmin=0)
#plt.imshow(density_3, cmap='jet', vmin=0)
#plt.colorbar()
#plt.show()

