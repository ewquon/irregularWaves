#!/usr/local/bin/python
import numpy as np
import matplotlib.pyplot as plt
import peaks # for estimating Hs
#from scipy import integrate
from scipy import optimize
from numpy import fft
from progressbar import ProgressBar, Bar, Percentage
np.seterr(divide='ignore')
pi = np.pi
twopi = 2.0*pi

#############################################
# INPUTS
verbose = True
debug = False
TOL = 1e-14

g  = 9.81
d  = 70.0           # water depth, m
Tp = 15.1           # peak period, s
Hs = 9.0            # significant wave height, m
wm = twopi/Tp       # modal frequency, rad/s
dt = 0.25           # simulation time-step size, s
Tmax = 3*60*60.0    # simulation length, s

Nwavelets = 2000
Nrealizations = 10 #500
use_equal_dw = True
plot_realizations = False       # should set Nrealizations=1 for this to be useful
plot_overlay = True            # only active if plot_realizations==True
optimize_panel_areas = False
#Nt = 1000 # number of points to calculate in each realization for estimating Hs and FFT calc
# can calculate from Tmax and dt

def wave_spectrum(w):    # Bretschneider
    wnorm4 = (w/wm)**4
    return 1.25/4.0 / (w*wnorm4) * Hs**2 * np.exp(-1.25/wnorm4) #m^2/(rad/s)

#############################################

#Tmax = Nwavelets*twopi/(wmax-wmin) # time after which the wave train is repeated (Baekkedal 2014, p.31)
#Tmax = Nwavelets*twopi/ wmax # limit as wmin->0; matches "sampling theorem" from Duarte 2014, p.2
deltaw = twopi/Tmax
wmin = deltaw
wmax = Nwavelets*deltaw
if wmax < 2*wm:
    print 'WARNING wmax/wm =',wmax/wm

if verbose: 
    print 'INPUT Tp             =',Tp,'s'
    print 'INPUT Hs             =',Hs,'m'
    print 'INPUT Tmax           =',Tmax,'s'
    print 'Calculated wm        =',wm,'rad/s'
    print 'Calculated wmin/max  =',wmin,wmax,'rad/s'
    print '--------------------------------------'

# OLD equal dw scheme
w_equal = np.linspace(wmin,wmax,Nwavelets)
S_equal = wave_spectrum(w_equal)
dw_equal = w_equal[1]-w_equal[0]
M0_equal = np.sum(S_equal)*dw_equal # zeroth moment a.k.a. spectral variance = sigma^2

# integrate to get "exact" area under spectral density curve
#M0 = integrate.romberg(wave_spectrum, wmin, wmax, show=verbose, tol=TOL)
# - analytical expression obtained by directly integrating the wave spectrum equation
totalarea = Hs**2/16.0 * (np.exp(-1.25*wm**4/wmax**4) - np.exp(-1.25*wm**4/wmin**4))
panelarea = totalarea / Nwavelets

# calculate panel edge locations
w0 = np.zeros((Nwavelets+1))
w0[0] = wmin
B = 1.25*wm**4
for i in range(Nwavelets):
    # obtained from integrated wave spectrum equation
    w0[i+1] = ( -B / np.log( 16/Hs**2*panelarea + np.exp(-B/w0[i]**4) ) )**0.25
wmaxerr = wmax - w0[-1]
w0[-1] = wmax

dw = np.diff(w0)
w = (w0[1:] + w0[:-1]) / 2.0
S = wave_spectrum( w )
M0 = S.dot( dw )

# DEBUG--use constant instead of variable dw
if use_equal_dw:
    assert( dw_equal == deltaw )
    dw = dw_equal*np.ones((Nwavelets))
    w  = w_equal
    S  = S_equal
    M0 = M0_equal

if verbose: 
    print ''
    print 'Exact variance                   =', totalarea
    print 'Hs from truncated spectrum       =', 4*totalarea**0.5
    print '  equal dw : calculated variance =', M0_equal
    print '  equal dw : estimated Hs        =', 4*M0_equal**0.5
    print '  equal A  : calculated variance =', M0
    print '  equal A  : estimated Hs        =', 4*M0**0.5
    print 'wmax err:',wmaxerr,' (accumulated numerical error in calculating panel edges)'
    print 'average/expected/stdev panel areas',np.mean(S*dw),panelarea,np.std(S*dw)

# DEBUG
#for Ai,err in zip(S*dw,S*dw-panelarea):
#    print Ai,err

#-------------------------------------------------------------------------------
#
# optimize panel areas to hopefully improve variance estimate
#
if optimize_panel_areas:# {{{
    if verbose: 
        print '\n--------------------------------------'
        print 'Optimizing panel areas...'
    #print '  before:',w0
# Optimizer actually seems to decrease the average area!
    def optF(wi):
        F = np.zeros((Nwavelets-1))
        w0 = np.zeros((Nwavelets+1))
        w0[0] = wmin
        w0[1:-1] = wi
        w0[-1] = wmax
        dw = np.diff(w0)
        Smid = wave_spectrum((w0[:-1]+w0[1:])/2.0)
        areas = Smid*dw
        delta_areas = np.diff(areas)
        #return delta_areas
        return delta_areas.dot( delta_areas )
    print 'initial residual:',optF(w0[1:-1])
    #result = optimize.leastsq( optF, w0[1:-1] )
    opt = {'disp':verbose, 'maxiter':200}
    result = optimize.minimize( optF, w0[1:-1], method='SLSQP', options=opt )
    print result.message
    w0[1:-1] = result.x
    #print result
    #for before,after in zip(w0[1:-1],result.x):
    #    print before,after

    dw = np.diff(w0)
    w = (w0[1:] + w0[:-1]) / 2.0
    S = wave_spectrum( w )
    print 'average/expected/stdev panel areas',np.mean(S*dw),panelarea,np.std(S*dw)# }}}

#-------------------------------------------------------------------------------
#
# generate and plot time history
#
# Bakkedal2014, eqn 10 :
# wave elevation, Z(x,t) = sum( A_i * cos(k_i*x - w_i*t + phi) )
#
fig, ax = plt.subplots(nrows=1)
fig.suptitle('Wave spectrum, {:d} components (Hs={:f}, Tp={:f})'.format(Nwavelets,Hs,Tp))

if plot_realizations or debug:
    plt.figure()
    plt.xlabel('time [s]')
    plt.ylabel('wave elevation [m]')
#t = np.linspace(0,Tmax,Nt)
t = np.arange(0,Tmax+dt,dt)
Nt = len(t)
dt = t[1]-t[0]
freq = fft.fftfreq(Nt,dt) * 2*pi #--fftfreq outputs cycles/time
df = freq[1]-freq[0]
Z = np.zeros((len(t)))
Hs_mean = 0.
A_max_mean = 0.
A_max_abs = 0.
Sfft_mean = np.zeros((Nt))
M0_rand = np.zeros((Nrealizations))
rand_A2 = np.zeros((Nwavelets))

if verbose: print '\n--------------------------------------'
if debug:
    print 'Generating random amplitude realizations'
else:
    pbar = ProgressBar(
            widgets=['Generating {:d} random amplitude realizations '.format(Nrealizations),Percentage(),Bar()],
            maxval=Nrealizations).start()
#print S*dw
for irand in range(Nrealizations):

    # generate random amplitude components
    sigma_R = np.sqrt(2/pi) * 2*S*dw
    rand_A2[:] = 0.0
    for i,sig in enumerate(sigma_R):
        if sig==0:
            rand_A2[i] = 0.0
            continue
        rand_A2[i] = np.random.rayleigh(scale=sig)
    rand_A = rand_A2**0.5
    M0_rand[irand] = np.sum(rand_A2)/2 #--eqn 97

    # generate random phase components
    rand_phi = np.random.random(Nwavelets) * twopi

    # calculate elevation profile
    if debug:
        for i in range(Nwavelets):
            print 'wavelet {:d}: w={:f}, A={:f}, phi={:f}'.format(i+1,w[i],rand_A[i],rand_phi[i])
    for i,ti in enumerate(t):
        Z[i] = np.sum( rand_A*np.cos(-w*ti + rand_phi) )
    if plot_realizations or debug: plt.plot(t,Z)

    # find peaks to estimate Hs
    Zpeaks,ipeaks = peaks.find_peaks(Z,t,Nsmoo=2)
    imax = np.nonzero( Zpeaks > 0 )
    maxima = Zpeaks[imax]
    if debug: plt.plot(t[ipeaks[imax]],maxima,'ro')
    maxima.sort()
    Hs_calc = 2*np.mean( maxima[-len(maxima)/3:] )
    if debug: print 'realization',irand+1,':  Hs =',Hs_calc
    Hs_mean += Hs_calc
    zmax = np.max(Z)
    A_max_mean += zmax
    A_max_abs = max(A_max_abs,zmax)

    # perform FFT to verify recovery of spectral content
    F = fft.fft(Z)/Nt
    P = np.abs(F)**2
    Sfft = P/df
    if debug: ax0.plot(freq[:Nt/2],Sfft[:Nt/2],label='realization {:d}'.format(irand+1))
    Sfft_mean += Sfft

    # DEBUG--overlay repeated intervals
    if plot_realizations and plot_overlay:
        #t2= np.linspace(  Tmax,2*Tmax,Nt) # DEBUG
        #t3= np.linspace(2*Tmax,3*Tmax,Nt) # DEBUG
        t2 = np.arange(  Tmax,2*Tmax+dt,dt)
        t3 = np.arange(2*Tmax,3*Tmax+dt,dt)
        Z2 = np.zeros((Nt))
        Z3 = np.zeros((Nt))
        for i,ti in enumerate(t2):
            Z2[i] = np.sum( rand_A*np.cos(-w*ti + rand_phi) )
        if plot_realizations or debug: plt.plot(t2-t2[0],Z2)
        print np.max(np.abs(Z2-Z))
        for i,ti in enumerate(t3):
            Z3[i] = np.sum( rand_A*np.cos(-w*ti + rand_phi) )
        if plot_realizations or debug: plt.plot(t3-t3[0],Z3)
        print np.max(np.abs(Z3-Z))

    if not debug: pbar.update(irand+1)
if not debug: pbar.finish()
Hs_mean /= Nrealizations
A_max_mean /= Nrealizations
Sfft_mean /= Nrealizations

if verbose:
    print '  absolute max amp.  =', A_max_abs
    print '  average max amp.   =', A_max_mean
    print '  average Hs         =', Hs_mean
    print '  average M0         =', np.mean(M0_rand) # for one-sided spectrum
    print '  variance of M0     =', np.var(M0_rand) # for one-sided spectrum
    print '  estimated Hs(M0)   =', 4*np.mean(M0_rand)**0.5

#-------------------------------------------------------------------------------
#
# plot spectrum
#
plotfn = ax.plot #ax.loglog
wplot = np.linspace(wmin,wmax,1000)
plotfn(wplot,wave_spectrum(wplot),'k-',linewidth=2,label='B-S')

plotfn(w_equal,S_equal,'ro',markersize=4,mfc='r',mec='r')

# plot panel outlines
plotfn(w,S,'ks')
for wi in w0[1:-1]: plotfn([wi,wi],[0,wave_spectrum(wi)],'0.8')

# TODO check: since we threw out the negative-frequency part of spectrum, we lost half the power in the spectrum...
#             thus the factor of two?
plotfn(freq[:Nt/2],2*Sfft_mean[:Nt/2],'-o',label='avg over {:d} realizations (equal dw)'.format(Nrealizations))
ax.set_xlim((wmin,wmax))
#ax.set_ylim((TOL,100))
ax.set_xlabel('w [rad/s]')
ax.set_ylabel('S [m^2/(rad/s)]')
ax.legend(loc='best')

#-------------------------------------------------------------------------------
# 
# plot sea snapshots
#
# Bakkedal2014, eqn 10 :
# wave elevation, Z(x,t) = sum( A_i * cos(k_i*x - w_i*t + phi) )
#
#Nsnapshots = 5
#t0 = -Tp
#Np = 2
#x = np.linspace(-500,500,501)
#t = np.arange(Nsnapshots,dtype=float)/(Nsnapshots-1) * Np*Tp
#Z = np.zeros((len(x)))
#fig2, ax_snapshot = plt.subplots(nrows=Nsnapshots,sharex=True)
#for itime in range(Nsnapshots):
#    ti = t[itime] + t0
#    Z[:] = 0.0
#    for i,xi in enumerate(x):
#        Z[i] = np.sum( A*np.cos(k*xi - w*ti) )
#    ax_snapshot[itime].plot(x,Z)
#    ax_snapshot[itime].set_title('t={:f}'.format(ti))
#    ax_snapshot[itime].set_ylim((-1.5*Hs,1.5*Hs))



plt.show()

