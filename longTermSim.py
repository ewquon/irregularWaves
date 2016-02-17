#!/usr/local/bin/python
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate # for calculating bandwidth
import peaks # for estimating Hs
from numpy import fft
from progressbar import ProgressBar, Bar, Percentage
np.seterr(divide='ignore')
pi = np.pi
twopi = 2.0*pi

#############################################
# INPUTS
verbose = True
debug = False
use_optimize = False # use scipy solver rather than hard-coded derivative
g  = 9.81
d  = 70.0           # water depth, m
Tp = 15.1           # peak period, s
Hs = 9.0            # significant wave height, m
wm = twopi/Tp       # modal frequency, rad/s
w0 = wm/2           # smallest frequency
w1 = 10*wm          # largest frequency

Nwavelets = 200 #100 is not enough to recover original B-S averaging over >1000 realizations
Nrealizations = 1000
plot_realizations = False
Nt = 500 # number of points in each realization

TOL = 1e-14
MAXITER = 10
#############################################
if use_optimize: from scipy import optimize

if verbose: 
    print 'INPUT Tp         =',Tp
    print 'INPUT Hs         =',Hs
    print 'Calculated wm    =',wm
    print '--------------------------------------'

def BSspectrum(w):
    wnorm4 = (w/wm)**4
    return 1.25/4.0 / (w*wnorm4) * Hs**2 * np.exp(-1.25/wnorm4) #m^2/(rad/s)

#w = np.linspace(0,5*wm,101)
#w = w[1:]
w = np.linspace(w0,w1,Nwavelets)
S = BSspectrum(w)
Sdb = 20*np.log10(S/np.max(S))
dw = w[1]-w[0]
A = np.sqrt(2*S*dw)

Tmax = Nwavelets*twopi/(w1-w0) # time after which the wave train is repeated

# - sanity check from Techet 2005 MIT notes, for a narrow-banded spectrum
M0 = np.sum(S)*dw # zeroth moment a.k.a. spectral variance = sigma^2
if verbose:
    print 'estimated Hs from spectrum:',4*M0**0.5

# - clean up Sdb so that UnivariateSpline doesn't fail
iInf = np.nonzero(np.abs(Sdb)==np.inf)[0]
if len(iInf) > 0: ist = iInf[-1]+1
else: ist = 0
w = w[ist:]
S = S[ist:]
A = A[ist:]
Sdb = Sdb[ist:]
assert( len(w) == Nwavelets )

# - calculate wavenumbers
k = np.zeros((len(w)))
kguess = w**2/g
if debug: print 'DEBUG: calculating wavenumbers'
for i,wi in enumerate(w):
    if use_optimize or debug:
        print 'Using Scipy solver to calculate wavenumbers'
        def dispersion_k(k): return k*np.tanh(k*d) - wi**2/g
        k[i] = optimize.newton( dispersion_k, kguess[i] )
    def newton_step(k):
        kd = k*d
        tkd = np.tanh(kd)
        return k*tkd - wi**2/g, kd*(1-tkd**2) + tkd # F(k), dF/dk
    kexactJ = kguess[i]
    niter = 0
    F,J = newton_step(kexactJ)
    if debug: print '  iter',niter,'resid',F
    while niter < MAXITER and np.abs(F) > TOL:
        kexactJ += -F/J
        niter += 1
        F,J = newton_step(kexactJ)
        if debug: print '  iter',niter,'resid',F
    if niter >= MAXITER: print 'WARNING: max newton iterations reached!'
    if debug: 
        if niter < MAXITER: print 'newton iteration converged in',niter,'steps'
        print 'diff btwn optimize and exact derivative calculation:',kexactJ-k[i]
        print i,wi,kguess[i],k[i],kexactJ,dispersion_k(k[i])
    k[i] = kexactJ

#-------------------------------------------------------------------------------
#
# plot spectrum
#
fig0, (ax0,ax1) = plt.subplots(nrows=2,sharex=True)
fig0.suptitle('Bretschneider spectrum (Hs={:f}, Tp={:f})'.format(Hs,Tp))

ax0.plot(w,S,'k-',linewidth=2,label='B-S')
Srange = ax0.axis()[2:]
ax0.set_ylabel('S [m^2/(rad/s)]')

#plt.semilogy(w,S) #--scaling is terrible!
ax1.plot(w,Sdb,'k-')
ax1.set_ylim((-55,0))
ax1.set_ylabel('20 log(S/Smax) [dB]')
ax1.set_xlabel('w [rad/s]')

# - find bandwidth from -3dB criterion
Sdb_3dB_fn = interpolate.UnivariateSpline(w,Sdb+3.0,s=0) #find roots at -3 dB;  s=0: no smoothing
wLH_3dB = Sdb_3dB_fn.roots()
assert(len(wLH_3dB)==2)
bw_3dB = np.diff(wLH_3dB)[0]
print 'low/high frequency at -3 dB:',wLH_3dB
ax1.plot(wLH_3dB,Sdb_3dB_fn(wLH_3dB)-3.0,'bs',label='-3dB')
ax0.plot([wLH_3dB[0],wLH_3dB[0]],Srange,'k--')
ax0.plot([wLH_3dB[1],wLH_3dB[1]],Srange,'k--')

# - find bandwidth from full width at half maximum (FWHM) criterion
S_FWHM_fn = interpolate.UnivariateSpline(w,S-0.5*np.max(S),s=0)
wLH = S_FWHM_fn.roots()
assert(len(wLH)==2)
bw = np.diff(wLH)[0]
print 'low/high frequency at half maximum:',wLH
ax1.plot(wLH,Sdb_3dB_fn(wLH)-3.0,'r^',label='FWHM')
ax0.plot([wLH[0],wLH[0]],Srange,'k-.')
ax0.plot([wLH[1],wLH[1]],Srange,'k-.')

print '-3 dB bandwidth:',bw_3dB
print 'FWHM:',bw
ax0.legend(loc='best')
ax1.legend(loc='best')

if verbose: print '======================================'
#-------------------------------------------------------------------------------
#
# generate and plot time history
#
# Bakkedal2014, eqn 10 :
# wave elevation, Z(x,t) = sum( A_i * cos(k_i*x - w_i*t + phi) )
#
fig1, (ax2,ax3) = plt.subplots(nrows=2,sharex=True)
ax2.set_ylabel('amplitude [m]')
ax3.set_ylabel('phase [deg]')
ax3.set_xlabel('frequency [rad/s]')

# - uniform phase shift = 0
#plt.figure()
#t = np.linspace(0,0.5*Tmax,500)
#Z = np.zeros((len(t)))
#for i,ti in enumerate(t):
#    Z[i] = np.sum( A*np.cos(-w*ti) )
#plt.plot(t,Z)
#plt.xlabel('time [s]')
#plt.ylabel('wave elevation [m]')

# - random phase scheme (ref Baekkedal 2014, p.32)
if plot_realizations or debug:
    plt.figure()
    plt.xlabel('time [s]')
    plt.ylabel('wave elevation [m]')
t = np.linspace(0,Tmax,Nt)
Z = np.zeros((len(t)))
Hs_mean = 0.
if debug: 
    print 'Generating random phase realizations'
else:
    pbar = ProgressBar(widgets=['Generating random phase realizations ',Percentage(),Bar()],maxval=Nrealizations).start()
for irand in range(Nrealizations):
    # generate random phases
    rand_phi = np.random.random(Nwavelets) * twopi
    ax3.plot(w,rand_phi*180./pi)

    # calculate elevation profile
    for i,ti in enumerate(t):
        Z[i] = np.sum( A*np.cos(-w*ti + rand_phi) )
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

    if not debug: pbar.update(irand+1)
if not debug: pbar.finish()
Hs_mean /= Nrealizations
print '  average Hs     =',Hs_mean
print '  variance (M0)  =',2*M0 # multiply by two since variance is the result of integrating from -inf to inf
    
# - random amplitude scheme (ref Baekkedal 2014, p.32)
if plot_realizations or debug:
    plt.figure()
    plt.xlabel('time [s]')
    plt.ylabel('wave elevation [m]')
t = np.linspace(0,Tmax,Nt)
dt = t[1]-t[0]
freq = fft.fftfreq(Nt,dt) * 2*pi #--fftfreq outputs cycles/time
Z = np.zeros((len(t)))
Hs_mean = 0.
Sfft_mean = np.zeros((Nt))
M0_rand = np.zeros((Nrealizations))
rand_A2 = np.zeros((Nwavelets))
if debug:
    print 'Generating random amplitude realizations'
else:
    pbar = ProgressBar(widgets=['Generating random amplitude realizations ',Percentage(),Bar()],maxval=Nrealizations).start()
for irand in range(Nrealizations):
    # generate random amplitude components
    sigma_R = np.sqrt(2/pi) * 2*S*dw
    rand_A2[:] = 0.0
    for i,sig in enumerate(sigma_R):
        rand_A2[i] = np.random.rayleigh(scale=sig)
    rand_A = rand_A2**0.5
    ax2.plot(w,rand_A)
    M0_rand[irand] = np.sum(rand_A2)/2 # sum( A2/(2*dw) ) * dw

    # generate random phase components
    rand_phi = np.random.random(Nwavelets) * twopi

    # calculate elevation profile
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
    df = freq[1]-freq[0]
    Sfft = P/df
    if debug: ax0.plot(freq[:Nt/2],Sfft[:Nt/2],label='realization {:d}'.format(irand+1))
    Sfft_mean += Sfft

    if not debug: pbar.update(irand+1)
if not debug: pbar.finish()
Hs_mean /= Nrealizations
Sfft_mean /= Nrealizations
print '  average Hs         =',Hs_mean
print '  average M0         =',2*np.mean(M0_rand)
print '  variance of M0     =',4*np.var(M0_rand)
print '  estimated Hs(M0)   =',4*np.mean(M0_rand)**0.5
    

# - plot these last so it ends up on top
#ax0.plot(freq[:Nt/2],Sfft_mean[:Nt/2],label='average over {:d} realizations'.format(Nrealizations))
# TODO check: since we threw out the negative-frequency part of spectrum, we lost half the power in the spectrum...?
ax0.plot(freq[:Nt/2],2*Sfft_mean[:Nt/2],label='average over {:d} realizations'.format(Nrealizations))
ax0.set_xlim((0,np.max(w)))
ax0.legend(loc='best')
ax2.plot(w,A,'k',linewidth=2)
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

