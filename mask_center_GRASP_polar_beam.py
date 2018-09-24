'''
read in grasp .grd file in polar form
mask central pixel (or 2 or 3) with zeros or average value.
Assumes 361 points in phi, 181 in theta.

Sept 2018.
Karl Young


'''

import numpy as np
import matplotlib.pylab as plt
import load_funcs as lf
import plot_funcs as pf
import pdb



def mask_center(grid,rings=1,infill=True):
  # assumes 361 in phi, 181 in theta
  # matrix from lf.load_grid()
  # rings: is number of phi rings to mask
  # infill: True replaces center ring with vaules from first kept outer ring.  False sets all to zero.

  unmask_ring = grid[rings,:]
  if infill:
    for i in np.arange(rings)+1:
	  grid[:i,:] = unmask_ring
  else:
    grid[:rings,:] = 0.0 + 0.0j

  return grid
  
#function to write data section of grasp .grd file
# input arrays must be complex

def write_grasp_grd(copol,xpol,fout):
  # convert to 1 D arrays
  copol = np.reshape(copol,(-1,))
  xpol =  np.reshape(xpol,(-1,))
  
  # seperate real, imaginary
  out = np.array([np.real(copol),np.imag(copol),np.real(xpol),np.imag(xpol)]).T
  
  # write as columns.
  with open(fout,'ab') as f:
    np.savetxt(f,out,delimiter=' ', fmt='%.10E')
	
  return

# copy header component to existing .grd file.
# read them from a .grd file
def read_write_grasp_header(fin, fout, headlines=11):
  with open(fout,'w') as outfile:
    with open(fin,'r') as infile:
	  for i in range(headlines): 
	    outfile.write(infile.readline())
  return

#write_grasp(cocomx,xcomx,'combined_grasp_beam_xfeed.grd')
#write_grasp(cocomy,xcomy,'combined_grasp_beam_yfeed.grd')


# load a file
# fname = 
fnames = ['4pi_beam_xfeed_center_150GHz.grd', '4pi_beam_yfeed_center_150GHz.grd']
for fname in fnames:

  fname_out = fname.split('.')[0]+'_mask2ring.grd'
  comain, xmain, arrmain = lf.load_grid(fname)
  # matrices are 181 rows, 361 colums. theta, phi on 1 deg grid.

  mask_co = mask_center(comain, rings=2,infill=True)
  mask_x = mask_center(xmain, rings=2,infill=True)

  read_write_grasp_header(fname,fname_out,headlines=11)
  write_grasp_grd(mask_co,mask_x, fname_out)





