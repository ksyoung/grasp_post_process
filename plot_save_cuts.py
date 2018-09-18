'''
Scratch code to plot and save sidelobe cuts.
March 2018.

use half main only, half stop+main

add sunshield and moon, earth positions to cuts.

'''

import numpy as np
import matplotlib.pylab as plt
import load_funcs as lf
import plot_funcs as pf
import pdb

# files, all full sky grid on 1 deg.
fmain_x = '../grasp10/PICO/Job_09/center_Feed_x_4pi_1deg_grid_mainonly.grd'
fstop_x = '../grasp10/PICO/Job_09/center_Feed_x_4pi_1deg_grid_stoponly.grd'
fmain_y = '../grasp10/PICO/Job_16/center_Feed_y_4pi_1deg_grid_mainonly.grd'
fstop_y = '../grasp10/PICO/Job_16/center_Feed_y_4pi_1deg_grid_stoponly.grd'

# load all data
comainx,xmainx,arrmainx = lf.load_grid(fmain_x)
costopx,xstopx,arrstopx = lf.load_grid(fstop_x)
comainy,xmainy,arrmainy = lf.load_grid(fmain_y)
costopy,xstopy,arrstopy = lf.load_grid(fstop_y)

# make right full sky grid. where is split to use front or back of stop!!?!??!
# if on + z side of theta + cos(phi)*78 then use main, if below use main+stop.

# assume all are 181 x 361 pnts, theta x phi.
def combine_grids(main,stop):
    combo=np.zeros((181,361)) + 0j

    for i in range(181):
        for j in range(361):
            if i > (90+np.cos(np.deg2rad(j))*78): # use main+stop
                combo[i,j] = main[i,j]+stop[i,j]
                #print 'theta: ',i
                #print 'phi: ',j
            else:  # use main only
                combo[i,j] = main[i,j]
    return combo

# run all through combo
cocomx = combine_grids(comainx,costopx)
cocomy = combine_grids(comainy,costopy)
xcomx  = combine_grids(xmainx,xstopx)
xcomy  = combine_grids(xmainy,xstopy)

# rewrite grasp files for sending to NERSC sims. write no header for now, just copy paste that on.
'''
def write_grasp(copol,xpol,fout):
  # convert to 1 D arrays
  copol = np.reshape(copol,(-1,))
  xpol =  np.reshape(xpol,(-1,))
  
  # seperate real, imaginary
  out = np.array([np.real(copol),np.imag(copol),np.real(xpol),np.imag(xpol)]).T
  
  # write as columns.
  np.savetxt(fout,out,delimiter=' ', fmt='%.10E')
  return

write_grasp(cocomx,xcomx,'combined_grasp_beam_xfeed.grd')
write_grasp(cocomy,xcomy,'combined_grasp_beam_yfeed.grd')

pdb.set_trace()
'''

# plot all on one fig for now.
fig1,ax1 = plt.subplots() # phi 0
fig2,ax2 = plt.subplots() # phi 45
fig3,ax3 = plt.subplots() # phi 90

ax1.set_ylim([-50,70])
ax2.set_ylim([-50,70])
ax3.set_ylim([-50,70])

# phis = [0,45,90] # phis to plot, need to be on 0-179
# hardcode phis of 0 , 45, 90 for now.
# do x,y pols at same cut on 1 plot.
for i,fig in enumerate([fig1,fig2,fig3]):
    phi=[45*i]
    pf.theta_phi_cuts(cocomx, xcomx, arrmainx, phi, fig, label=r'pol_x $\phi=%g$' %phi[0],verbose=False, save=False)
    pf.theta_phi_cuts(xcomy, cocomy, arrmainy, phi, fig, label='pol_y $\phi=%g$' %phi[0],verbose=False, save=False)

ax1.legend()
ax2.legend()
ax3.legend()

# draw earth, moon, and sunshields on here too.
# for phi = 0, angles are:
  # shields, theta = 180, 40  (20 at phi = 90)
  # earth, theta =  45, 175 (assuming herschel orbit)
  # moon, theta =  35, 165

ax1.axvline(x=38,color='brown',alpha=.7, label='sunshield')
ax1.axvline(x=-179,color='brown',alpha=.7)
ax3.axvline(x=70,color='brown',alpha=.7, label='sunshield')
ax3.axvline(x=-70,color='brown',alpha=.7)
#earth, moon
#ax1.axvline(x=35,color='grey', label='moon')
#ax1.axvline(x=-165,color='grey')
#ax1.axvline(x=45,color='tab:green', label='earth')
#ax1.axvline(x=-175,color='tab:green')

ax1.legend()
ax2.legend()
ax3.legend()



# plot 4 pi beams as well.
if True:
    fig4,fig5,fig6 = pf.plot_fullsky_grid(cocomx,xcomx,arrmainx,dbmax=65.7,
                                          fname='pol x', verbose=False, save=False)
    fig7,fig8,fig9 = pf.plot_fullsky_grid(cocomy,xcomy,arrmainy,dbmax=65.7,
                                          fname='pol y', verbose=False, save=False)

    fig4.savefig('./output/polx_corrected_4pi.png')
    fig8.savefig('./output/poly_corrected_4pi.png')
pdb.set_trace()
#combo = combine_grids(comainx,comainx)
#pf.plot_fullsky_grid(cocomx,xcomx,arrmainx,dbmax=65.7)

fig1.savefig('./output/polxy_corrected_cuts_phi_0.png')
fig2.savefig('./output/polxy_corrected_cuts_phi_45.png')
fig3.savefig('./output/polxy_corrected_cuts_phi_90.png')
