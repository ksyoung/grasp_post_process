'''
Code to plot 4 pi beams from GRASP in python.
1st pass using python's polar projection.
2nd pass try python's mollweide projection
3rd pass do it in healpix?




Karl Young, UMN
Jan 2018
'''

import matplotlib.pylab as plt
import numpy as np
import load_funcs as lf
import sys

import pdb



#Do command line args later.  # or put this in function form.


def plot_main_beam_grid(fname,verbose=True,save=True):
    '''
    load and Plot grid from file.  Assumes small grid centered on main beam. 
    Only does uv-grid for now.
    input is file name.
    saves/shows plots.
    output fig handles,  
    '''
    # load file
    copol, xpol, [nx,ny,dx,dy,xcen,ycen, grid_type] = lf.load_grid(fname)

    dBcopol = lf.field2amp_dB(copol)
    dBxpol = lf.field2amp_dB(xpol)

    fig1, ax1 = plt.subplots(1)  #co pol
    fig2, ax2 = plt.subplots(1)  # x pol
    
    if grid_type == 'uv-grid':
        # make meshgrid
        u_vect = np.linspace(xcen-(nx-1)/2*dx,xcen+(nx-1)/2*dx,nx)
        v_vect = np.linspace(ycen-(ny-1)/2*dy,ycen+(ny-1)/2*dy,ny)
        u_grid, v_grid = np.meshgrid(u_vect,v_vect)

        # plot
        cax1 = ax1.pcolormesh(u_grid, v_grid, dBcopol, cmap='inferno') #set z range with vmin=, vmax=
        ax1.set_aspect((u_vect[0]-u_vect[-1])/(v_vect[0]-v_vect[-1])) # make pixels sqaure. I think this is right even if len(u) != len(v)
        cax2 = ax2.pcolormesh(u_grid, v_grid, dBxpol, cmap='inferno')
        ax2.set_aspect((u_vect[0]-u_vect[-1])/(v_vect[0]-v_vect[-1]))

        ax1.set_xlabel('u')
        ax1.set_ylabel('v')
        ax2.set_xlabel('u')
        ax2.set_ylabel('v')

        cbar1 = fig1.colorbar(cax1)
        cbar1.set_label('Amplitude, dB', rotation=270)
        ax1.set_title('co-pol, %s' %file.split('/')[-1])

        cbar2 = fig2.colorbar(cax2)
        cbar2.set_label('Amplitude, dB', rotation=270)
        ax2.set_title('X-pol, %s' %file.split('/')[-1])
    else:
        print 'can only do u-v grids as of now. Sorry.'
        sys.exit()

    if save:
        fig1.savefig('uv-copol_%s.png' %fname.split('/')[-1][:-4])
        fig2.savefig('uv-xpol_%s.png' %fname.split('/')[-1][:-4])

    if verbose:
        print 'Colorbar scale is: %.2f to %.2f' %(dBmax, dBmin)
        plt.show()
    
    return fig1, fig2


def plot_fullsky_grid(copol, xpol, param_arr,
                      fname='nameless',verbose=True,save=True,dbmax=np.nan):
    '''
    Plot grid from copol/xpol data array.
    only does theta-phi grid for now.
    input is: 
      data in copol, xpol, [array] format output by lf.load_grid.
      fname=, name for plots and such.
      verbose=T/F, talk or not.
      save=T/F, save plots as fname[:-4].png
      dbmax=, upper cutoff of db plot range. lower is upper - 120.
    no output, just saves/shows plots.
    '''

    ## load file

    # unpack lf.load_grid() parameters
    [nphi,nth,dphi,dth,phist,phien,thst,then, grid_type] = param_arr

    dBcopol = lf.field2amp_dB(copol)
    dBxpol = lf.field2amp_dB(xpol)

    if dbmax is np.nan:
        dbmax = np.max(dBcopol)
    dbmin = dbmax-120

    if grid_type == 'Theta-Phi':

        fig1, (ax1a,ax1b) = plt.subplots(1,2,subplot_kw={'projection': 'polar'},
                                         figsize=(8,4))  #co pol
        fig2, (ax2a,ax2b) = plt.subplots(1,2,subplot_kw={'projection': 'polar'},
                                         figsize=(8,4))  # x pol
        fig3, (ax3a,ax3b) = plt.subplots(2,1,figsize=(6,4))  # make a sqaure project to check for craziness.

        # make meshgrid
        half_theta_points = nth/2+1
        phi_vect = np.deg2rad(np.linspace(phist,phien,nphi)) # polar plot seems to like radians.
        th_vect = np.linspace(thst,then,nth) # should be 0 to 180
        th_vecta = (np.linspace(0,90,half_theta_points))
        th_vectb = (np.linspace(90,0,half_theta_points))
        phi_grid, th_grid = np.meshgrid(np.rad2deg(phi_vect),th_vect) # i like degrees better.
        phi_grida, th_grida = np.meshgrid(phi_vect,th_vecta)
        phi_gridb, th_gridb = np.meshgrid(phi_vect,th_vectb)

        # plot
        cax1 = ax1a.pcolormesh(phi_grida, th_grida, dBcopol[:(half_theta_points),:], cmap='inferno', vmin=dbmin, vmax=dbmax)
        cax1 = ax1b.pcolormesh(phi_gridb, th_gridb, dBcopol[-(half_theta_points):,:], cmap='inferno', vmin=dbmin, vmax=dbmax)
        #set z range with vmin=, vmax=
        #ax1.set_aspect((u_vect[0]-u_vect[-1])/(v_vect[0]-v_vect[-1])) # make pixels sqaure. I think this is right even if len(u) != len(v)
        cax2 = ax2a.pcolormesh(phi_grida, th_grida, dBxpol[:half_theta_points,:], cmap='inferno', vmin=dbmin, vmax=dbmax)
        cax2 = ax2b.pcolormesh(phi_gridb, th_gridb, dBxpol[-half_theta_points:,:], cmap='inferno', vmin=dbmin, vmax=dbmax)
        cax3 = ax3a.pcolormesh(phi_grid,th_grid,dBcopol, cmap='inferno', vmin=dbmin,vmax=dbmax)
        cax3 = ax3b.pcolormesh(phi_grid,th_grid,dBxpol,  cmap='inferno', vmin=dbmin,vmax=dbmax)
        ax3a.set_aspect(dth/dphi)
        ax3b.set_aspect(dth/dphi)

        # add labels etc.

        ax1a.set_xlabel('Forward 2Pi')
        ax1b.set_xlabel('Backward 2Pi')    
        #ax1a.set_ylabel('Theta')
        fig1.subplots_adjust(right=0.85)
        cbar_ax1 = fig1.add_axes([0.9, 0.15, 0.02, 0.7])
        cbar1 = fig1.colorbar(cax1, cax=cbar_ax1)
        cbar1.set_label('Amplitude, dB', rotation=270)
        fig1.suptitle('co-pol, %s' %fname.split('/')[-1])

        ax2a.set_xlabel('Foward 2Pi')
        ax2b.set_xlabel('Backward 2Pi')    
        #ax2a.set_ylabel('Theta')
        fig2.subplots_adjust(right=0.85)
        cbar_ax2 = fig2.add_axes([0.9, 0.15, 0.02, 0.7])    
        cbar2 = fig2.colorbar(cax2, cax=cbar_ax2)
        cbar2.set_label('Amplitude, dB', rotation=270)
        fig2.suptitle('X-pol, %s' %fname.split('/')[-1])   

        ax3b.set_xlabel('Phi (deg)')
        ax3a.set_ylabel('Theta (deg)')
        ax3b.set_ylabel('Theta (deg)')
        cbar_ax3 = fig3.add_axes([0.85, 0.15, 0.02, 0.7])    
        cbar3 = fig3.colorbar(cax3, cax=cbar_ax3)
        cbar3.set_label('Amplitude, dB', rotation=270)
        ax3a.set_title('co-pol (top), X-pol (lower),\n%s' %fname.split('/')[-1])   

    else:
        print 'Can only plot theta-phi grids currently'
        sys.exit()
    if save:
        fig1.savefig('./output/copol_%s.png'  %fname.split('/')[-1][:-4])
        fig2.savefig('./output/xpol_%s.png'  %fname.split('/')[-1][:-4])    
        fig3.savefig('./output/rect_projection_both_%s.png'  %fname.split('/')[-1][:-4])

    if verbose:
        print 'Colorbar scale is: %.2f to %.2f' %(dbmax, dbmin)
        fig1.show()
        fig2.show()
        fig3.show()
        
    return fig1, fig2, fig3


def theta_phi_cuts(copol, xpol, param_arr,
                   phis, fig, label='',verbose=True, save=True):
    '''
    make cuts through a full sky theta-phi map 
    inputs,
      copol and xpol data in format output from lf.load_grid()
      phi angles for cuts. must be entered as a list.
      figure handle
      label=, label for that data set.

    ouputs,
      fig handles
    '''

    ## load file
    #copol, xpol, [nx,ny,dx,dy,xcen,ycen, grid_type] = lf.load_grid(fname)
    ## note for theta, phi, X = phi. Y=theta

    # unpack lf.load_grid() parameters
    [nx,ny,dx,dy,xs,xe,ys,ye, grid_type] = param_arr

    dBcopol = lf.field2amp_dB(copol)
    dBxpol = lf.field2amp_dB(xpol)

    #initialize vector for cut. length = 2 len(theta). rows = # phis.
    copol_cuts = np.zeros([len(phis),2*ny])
    xpol_cuts = np.zeros([len(phis),2*ny])
    print 'cuts shape  ', np.shape(copol_cuts)  # should be (phis x 361) usually

    phi_vect = np.linspace(xs,xe,nx) #need for getting right position in copol/xpol matrices
    th_vect = np.linspace(180,-180,2*ny)  # maybe shouldn't hardcode this?

    # get beam data
    for i,phi in enumerate(phis):
        #find right column, is phi and phi+180.
        #put that in right row
        copol_cuts[i,:ny] = np.flipud(dBcopol[:,np.where(phi_vect==phi)[0][0]])
        copol_cuts[i,ny:] = dBcopol[:,np.where(phi_vect==(phi+180))[0][0]]
        xpol_cuts[i,:ny] = np.flipud(dBxpol[:,np.where(phi_vect==phi)[0][0]])
        xpol_cuts[i,ny:] = dBxpol[:,np.where(phi_vect==(phi+180))[0][0]]

    # plot matrix, each row is 1 cut.
    plt.figure(fig.number)
    plt.plot(th_vect, copol_cuts.T, label=label) #defined matrix wrong early on . . . .
    plt.xlabel('Theta, (degrees)')
    plt.ylabel('dB')
    #plt.show()

    return fig

def y_rot_mat(angle):
    # makes the 3x3 rotation matrix and returns it.
    # angle in degrees
    angle = np.deg2rad(angle)
    rot_mat = np.zeros((3,3))
    rot_mat[0,0] = np.cos(angle)
    rot_mat[2,2] = np.cos(angle)
    rot_mat[1,1] = 1.
    rot_mat[0,2] = np.sin(angle)
    rot_mat[2,0] = -np.sin(angle)    
    return rot_mat

def cart2sphere(xyz):
    # convert array, 1D len 3, to spherical
    # returns in degrees
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    
    phi = np.rad2deg(np.arctan2(y,x)) # returns -pi to pi, next line take modulo to put in on 0-360
    phi = phi % 360.  # maps onto 0-360 nicely. thanks stackoveflow!
    r = np.sqrt(x**2. + y**2. + z**2.)
    th = np.rad2deg(np.arccos((z/r))) # returns 0 to pi
    th_phi_r = [th,phi,r]
    return th_phi_r

def sphere2cart(th_phi_r):
    # convert array, 1D len 3, to cartesian
    theta = np.deg2rad(th_phi_r[0])
    phi = np.deg2rad(th_phi_r[1])
    r = th_phi_r[2]
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    xyz = [x,y,z]
    return xyz
    
def y_rot_sphere(th_phi_r,angle):
    # combo of 3 funchtions above. rotate about y axis in spherical coords.
    # convert to x-y
    xyz = sphere2cart(th_phi_r)
    # rotate to boresight frame?
    rot_mat = y_rot_mat(angle)
    xyz_new = np.matmul(rot_mat,xyz)
    
    # convert back to phi-theta
    th_phi_r_new = cart2sphere(xyz_new) # r should be 1.
    return th_phi_r_new

def moon_earth_view_ring(rL2,alpha,beta,fig):
    # put ring of moon and earth viewing angles on a plot
    # assume at worst point for all precession angles. traces a ring.
    # input:
    #  rL2, L2 orbit radius (km)
    #  alpha, beta : telescope angles to precession and spin axis. in degrees
    #  fig, figure handle for polar beam plot to put overlay on.

    dL2 = 1.5e6  #km
    rmoon = 3.844e5  #km
    N = 360  # number of pnts in line.
    
    mang = np.rad2deg(np.arctan2(dL2,rL2+rmoon))  # deg, worst case
    eang = np.rad2deg(np.arctan2(dL2,rL2))  # deg, worst case

    # get ring positions in satellite frame, precession axis = 0.
    # should be 1 theta, const phi.
    th_m_sat = mang+alpha
    th_e_sat = eang+alpha

    print th_m_sat, th_e_sat
    ## Moon
    # make vector of points.
    th_phi_r = np.ones((3,N))
    th_phi_r[0,:] = th_m_sat # theta
    th_phi_r[1,:] = np.linspace(0,360,N)  # phi
    # rotate!
    [theta_m_grasp,phi_m_grasp,r] = (y_rot_sphere(th_phi_r,beta)) 
    ## Earth
    # make vector of points.
    th_phi_r = np.ones((3,N))
    th_phi_r[0,:] = th_e_sat # theta
    th_phi_r[1,:] = np.linspace(0,360,N)  # phi
    # rotate!
    [theta_e_grasp,phi_e_grasp,r] = (y_rot_sphere(th_phi_r,beta)) 
    
    # needs to convert polar 
    ax = fig.axes[0]

    if ax.name == 'polar':
        ax1 = ax  #front 2pi
        ax2 = fig.axes[1]  # back 2pi

        # plotting  Needs to be made more complicated. so theta and phi on split between forward and backward 2pi.
        mask_m_fr = np.where(theta_m_grasp <= 90.)
        mask_e_fr = np.where(theta_e_grasp <= 90.)
        mask_m_ba = np.where(theta_m_grasp >= 90.)
        mask_e_ba = np.where(theta_e_grasp >= 90.)  

        
        # misusing polar a bit, so phi is x coord, theta is y.
        # polar plot needs first coord in radians. I dislike this greatly.
        ax1.plot(np.deg2rad(phi_m_grasp)[mask_m_fr], theta_m_grasp[mask_m_fr],'o',color='grey', markersize=1)
        ax1.plot(np.deg2rad(phi_e_grasp)[mask_e_fr], theta_e_grasp[mask_e_fr], 'o',color='tab:green', markersize=1)
        # back 2 pi
        ax2.plot(np.deg2rad(phi_m_grasp)[mask_m_ba], np.abs(theta_m_grasp[mask_m_ba]-180),'o',color='grey', markersize=1)
        ax2.plot(np.deg2rad(phi_e_grasp)[mask_e_ba], np.abs(theta_e_grasp[mask_e_ba]-180), 'o',color='tab:green', markersize=1)


        
        print 'plotting 2 polar plots'
    #
    else:
        ax.plot(phi_m_grasp, theta_m_grasp, '.',color='grey',ms=1)  
        ax.plot(phi_e_grasp, theta_e_grasp, '.',color='tab:green',ms=1)
        print 'plotting 1 rectangular plot.'
        
    # plotting for fun.
    '''
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[0][:],xyz[1][:],zs=xyz[2][:])
    ax.scatter(xyz_new[0][:],xyz_new[1][:],zs=xyz_new[2][:])

    fig2, ax2 = plt.subplots()
    ax2.scatter(theta,phi)

    pdb.set_trace()
    '''
    return fig
