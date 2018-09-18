'''
Quick script to calculate GRASP field angles (phi, theta) for new 
feed in focal plane.

inputs:
plate scale (deg/cm)
x_shift  (cm)  offset of new feed in X direction
y_shift  (cm)  offset of new feed in Y direction

outputs:
phi   (deg)  grasp angle of field at infinity (sign may be wrong)
theta (deg)  grasp angle of field at infinity (sign may be wrong)

'''

from sys import argv
import numpy as np

import pdb


# read in command line args
try:
  ps = float(argv[1])
  x_cm = float(argv[2])
  y_cm = float(argv[3])

except:
  print 'Error: wrong inputs \n \nRun as:  \n'+\
      'calc_field_angle.py <plate_scale (deg/cm)> <x_pos (cm)> <y_pos (cm)>\n'

# from f/#  !!
# ps = (180/pi) / ( D * f#)  is in deg / length.
  
  
# calc new angles
x_rad = np.deg2rad(ps*x_cm)
y_rad = np.deg2rad(ps*y_cm)

x_deg = (ps*x_cm)
y_deg = (ps*y_cm)

phi = np.rad2deg(np.arctan2(y_deg,-x_deg))
theta1 = y_deg/np.sin(np.deg2rad(phi))
theta2 = -x_deg/np.cos(np.deg2rad(phi))
print theta1, theta2
#pdb.set_trace()

if x_cm == 0:
  theta = theta1
else:
  theta = theta2

if phi < 0:
  phi = phi + 360

# output
print 'Angles calculated: Don\'t trust signs!'
print 'Theta: %.4f \nPhi: %.4f' %(theta,phi)

