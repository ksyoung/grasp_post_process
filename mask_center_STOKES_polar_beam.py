'''
Hacked up code to mask center pixels of a stokes parameter beam as output 
by planck LevelS function 'grasp2stokes'.
The data is read from a fits file with I, Q, U, V.
and is written back to an I, Q, U, V fits file of the same format (not working yet!)

region to mask is hardcoded in the .py file.

pixels are replaced with the average value of their 3 outer neighbors.

'''
import pyfits
import numpy as np
import copy
import sys

def mask_thetaphi_map(redat, leni,lenj):
    # i is theta dimension, 
    # j is phi dimension
    mask = copy.deepcopy(redat)
    for i in range(leni+1,-1,-1):
        for j in range(lenj):
            if j == 0:
                mask[i,j] = np.mean([mask[i+1,-1], mask[i+1,0],mask[i+1,1]])
            elif j == lenj-1:
                mask[i,j] = np.mean([mask[i+1,j-1], mask[i+1,j], mask[i+1,0]])
            else:
                mask[i,j] = np.mean(mask[i+1,(j-1):(j+2)])
    return mask


# file_map = '../../20180921_double_res_main_beam/4pi_beam_xfeed_center_150GHz_mainb_map.fits'

file_map = sys.argv[1]

mask_theta = 25


data, header = pyfits.getdata(file_map,1, header=True)
with pyfits.open(file_map) as hdu:
  header = hdu[1].header
  data = hdu[1].data
  
  Ntheta = header['Ntheta'] # pionts in theta
  Nphi =  header['Nphi']    # points in phi
  Npnts = header['Naxis2']   # total data points in 1 fits table column/filed/whatever it's called.
  
  
  redatI = np.reshape(data['Beamdata'],  (Ntheta,Nphi))
  redatQ = np.reshape(data['BeamdataQ'], (Ntheta,Nphi))
  redatU = np.reshape(data['BeamdataU'], (Ntheta,Nphi))
  redatV = np.reshape(data['BeamdataV'], (Ntheta,Nphi))
  
  
  maskI = mask_thetaphi_map(redatI, mask_theta, Nphi)
  maskQ = mask_thetaphi_map(redatQ, mask_theta, Nphi)
  maskU = mask_thetaphi_map(redatU, mask_theta, Nphi)
  maskV = mask_thetaphi_map(redatV, mask_theta, Nphi)
  
  
  data['Beamdata'] = np.reshape(maskI, Npnts)
  data['BeamdataQ'] = np.reshape(maskQ, Npnts)
  data['BeamdataU'] = np.reshape(maskU, Npnts)
  data['BeamdataV'] = np.reshape(maskV, Npnts)
  
  
  # write!
  new_file = file_map[:-5] + '_centermask_%gpx.fits' %mask_theta
  hdu.writeto(new_file)

