#!/usr/local/bin/python
import numpy as np
import matplotlib.pyplot as plt
import peaks # for estimating Hs
#from scipy import integrate
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
#wmin = wm/2         # smallest frequency, rad/s
wmax = 5*wm        # largest frequency, rad/s

Nwavelets = 30
Nrealizations = 1
plot_realizations = True
Nt = 1000 # number of points to calculate in each realization
def wave_spectrum(w):    # Bretschneider
    wnorm4 = (w/wm)**4
    return 1.25/4.0 / (w*wnorm4) * Hs**2 * np.exp(-1.25/wnorm4) #m^2/(rad/s)

wmin = wmax/Nwavelets
Tfinal = 3*60*60.0  # simulation length, s

#############################################

#Tmax = Nwavelets*twopi/(wmax-wmin) # time after which the wave train is repeated (Baekkedal 2014, p.31)
Tmax= Nwavelets*twopi/ wmax #limit as wmin->0

if verbose: 
    print 'INPUT Tp         =',Tp,'s'
    print 'INPUT Hs         =',Hs,'m'
    print 'Calculated wm    =',wm,'rad/s'
    print 'Calculated Tmax  =',Tmax,'s'
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

#dw = np.diff(w0)
#w = (w0[1:] + w0[:-1]) / 2.0
#S = wave_spectrum( w )
#M0 = S.dot( dw )

dw = dw_equal*np.ones((Nwavelets))
w  = w_equal
S  = S_equal
M0 = M0_equal

if verbose: 
    print ''
    print 'Hs from truncated spectrum       =', 4*totalarea**0.5
    print '  equal dw : calculated variance =', M0_equal
    print '  equal dw : estimated Hs        =', 4*M0_equal**0.5
    print '  equal A  : calculated variance =', M0
    print '  equal A  : estimated Hs        =', 4*M0**0.5
    print 'wmax err:',wmaxerr,' (accumulated numerical error in calculating panel edges)'
    print '--------------------------------------'
    print ''

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
t = np.linspace(0,Tmax,Nt)
dt = t[1]-t[0]
freq = fft.fftfreq(Nt,dt) * 2*pi #--fftfreq outputs cycles/time
df = freq[1]-freq[0]
if verbose: print 'dt =',dt,'s  df =',df,'rad/s'
Z = np.zeros((len(t)))
Hs_mean = 0.
Sfft_mean = np.zeros((Nt))
M0_rand = np.zeros((Nrealizations))
rand_A2 = np.zeros((Nwavelets))

if debug:
    print 'Generating random amplitude realizations'
else:
    pbar = ProgressBar(
            widgets=['Generating {:d} random amplitude realizations '.format(Nrealizations),Percentage(),Bar()],
            maxval=Nrealizations).start()
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
    if verbose:
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

    # perform FFT to verify recovery of spectral content
    F = fft.fft(Z)/Nt
    P = np.abs(F)**2
    Sfft = P/df
    if debug: ax0.plot(freq[:Nt/2],Sfft[:Nt/2],label='realization {:d}'.format(irand+1))
    Sfft_mean += Sfft

    # DEBUG--overlay repeated intervals
    t2= np.linspace(  Tmax,2*Tmax,Nt) # DEBUG
    t3= np.linspace(2*Tmax,3*Tmax,Nt) # DEBUG
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
Sfft_mean /= Nrealizations

if verbose:
    print '  average Hs         =',Hs_mean
    print '  average M0         =',np.mean(M0_rand) # for one-sided spectrum
    print '  variance of M0     =',np.var(M0_rand) # for one-sided spectrum
    print '  estimated Hs(M0)   =',4*np.mean(M0_rand)**0.5

#-------------------------------------------------------------------------------
#
# plot spectrum
#
plotfn = ax.plot #ax.loglog
wplot = np.linspace(wmin,wmax,1000)
plotfn(wplot,wave_spectrum(wplot),'k-',linewidth=2,label='B-S')

# plot panel outlines
plotfn(w,S,'ks')
for wi in w0: plotfn([wi,wi],[0,wave_spectrum(wi)],'0.8')

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

