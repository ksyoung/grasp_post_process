'''
Functions to load data from GRASP output files



Jan 24, 2018  Karl Young, UMN

'''

import numpy as np

import pdb


def load_grid(filepath, verbose=True):
  '''
  read in spherical beam grid file, *.grd
  get coordinate system from header, print.
  store in NxM array. Print limits

  inputs:
   filepath, path to file. :-D
   verbose, print infor on loaded array. 

  outputs
   copol, xpol, [nx,ny,dx,dy,xs,xe,ys,ye, grid_type[igrid]]
   ? copol and xpol are matrices with dB on a grid.  
  '''

  # initialize some variables
  n_plus = 0
  n_header = 0

  # and some data to ref later.
  grid_type = {1:'uv-grid',4:'El over Az', 5:'El and Az',
               6:'Az over El', 7:'Theta-Phi'}
  
  # get number of header rows
  with open(filepath, 'r') as file:
    for nrow,line in enumerate(file):
      #pdb.set_trace()
      if line == '++++\n':
        n_plus = nrow
        n_header = nrow + 6  # this 6 could be wrong if formats change.
      if nrow > 30:  break # avoid reading the entire file. 
 
    file.seek(0,0) # go back to reread header.    
    # get useful header data
    for nrow, line in enumerate(file):
      if nrow == (n_plus + 2):
        [nset, icomp, ncomp, igrid] = np.fromstring(line,dtype=int, sep=' ')
        line = next(file)
        [ix, iy] = np.fromstring(line, dtype=int, sep=' ') # may be different if nset is not 1?
        line = next(file)
        [xs, ys, xe, ye] = np.fromstring(line, dtype=float, sep=' ') #grid definitions
        line = next(file)
        [nx, ny, klimit] = np.fromstring(line, dtype=int, sep=' ') #N cols, rows.
        
      if nrow == n_header: break        

    #get and store text header
    file.seek(0,0)
    header = [next(file) for i in xrange(n_plus)]
      
  #print file parameters.
  if verbose:
    print 'Parameters of data, file is %s' %filepath.split('/')[-1]
    print '  Num beams : %g' %nset
    print '  Num fields: %g' %ncomp
    if ncomp==2: print '    Far-field'
    if icomp==3:
      print '    Linear, co-pol and X-pol' 
    else:
      print '!!!\nUnkown polarization definition\n!!!'
    print '  Grid type is: %s' %grid_type[igrid]

  # calc x,y grid parameters.
  dx = (xe-xs)/(nx-1)
  dy = (ye-ys)/(ny-1)

  # load all data
  raw_array = np.loadtxt(filepath, skiprows=n_header)

  # reparse into NxM arrays to match grid. Assume 1 beam with co-pol, x-pol
  copol = np.reshape(raw_array[:,0] + 1j*raw_array[:,1], (ny,nx)) # theta is y, phi is x so this should be (theta, phi).
  xpol = np.reshape(raw_array[:,2] + 1j*raw_array[:,3],  (ny,nx)) # as above, theta increasing wtih columns and phi increaseing with rows.  
  
  # make coord grids in plotting code.
  
  return copol, xpol, [nx,ny,dx,dy,xs,xe,ys,ye, grid_type[igrid]]
  
def make_grid():  
  '''
  Make a grid to use with plotting tools.
  Grid format is?    
  '''
  pass
  return


def field2amp_dB(array):
  # convert array of complex e-field to amplitude
  if 0 in np.abs(array):
    pos = np.abs(array)
    pos[np.where(pos==0.0)] = 2.5e-308  #just over smallest floating point num
    #pos[np.where(pos==0.0)] = 1e-10  # -200 db.
    return 20*np.log10(pos)
  return 20*np.log10(np.abs(array))



## plot a slice function. choose theta, phi, or rotation angle and generate a slice?  1 function for in theta, one for in u-v, various rotatoin options.  Currently plot functions in seperate file.. . . Go look there!

## do all plot commands go in functions? or other .py scripts?  hmmm..????
