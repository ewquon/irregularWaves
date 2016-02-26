#!/usr/local/bin/python
import numpy as np
import matplotlib.pyplot as plt
import peaks # for estimating Hs
#from scipy import integrate
from scipy import optimize
from numpy import fft
from progressbar import ProgressBar, Bar, Percentage
import time
np.seterr(divide='ignore')
pi = np.pi
twopi = 2.0*pi

#############################################
# INPUTS
verbose = True
debug = False
TOL = 1e-14
MAXITER = 25

g  = 9.81
d  = 70.0           # water depth, m
Tp = 15.1           # peak period, s
Hs = 9.0            # significant wave height, m
wm = twopi/Tp       # modal frequency, rad/s
Tmax = 3*60*60.0    # simulation length, s
Nwavelets = 6000
Nrealizations = 1 #500
#Nt = 5000 # number of points to calculate in each realization for estimating Hs and FFT calc
Nt = 2*Nwavelets
random_amplitude = False

write_coeffs = False
write_name = 'equalDw'

xref = 600.0 # Star wave surface reference location, m
xshift = 0.0 # spatial shift to accommodate the domain
tshift = 0.0 # temporal shift

plot_realizations = True       # should set Nrealizations=1 for this to be useful
plot_overlay = False            # only active if plot_realizations==True
optimize_panel_areas = False

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
    print '*** WARNING wmax/wm =',wmax/wm,'***'

if verbose: 
    print 'INPUT # wave components  =',Nwavelets
    print 'INPUT Tp                 =',Tp,'s'
    print 'INPUT Hs                 =',Hs,'m'
    print 'INPUT Tmax               =',Tmax,'s'
    print 'Calculated wm            =',wm,'rad/s'
    print 'Calculated wmin/max      =',wmin,wmax,'rad/s'
    print '--------------------------------------'

w = np.linspace(wmin,wmax,Nwavelets)
S = wave_spectrum(w)
dw = (w[1]-w[0]) * np.ones((Nwavelets))
M0 = S.dot(dw) # zeroth moment a.k.a. spectral variance = sigma^2

# integrate to get "exact" area under spectral density curve
#M0 = integrate.romberg(wave_spectrum, wmin, wmax, show=verbose, tol=TOL)
# - analytical expression obtained by directly integrating the wave spectrum equation
totalarea = Hs**2/16.0 * (np.exp(-1.25*wm**4/wmax**4) - np.exp(-1.25*wm**4/wmin**4))

if verbose: 
    print ''
    print 'Exact variance (wmax->Inf) =',Hs**2/16.0
    print 'Exact variance (wmax={:f}) = {:f}'.format(wmax,totalarea)
    print 'Hs from truncated spectrum       =', 4*totalarea**0.5
    print '  equal dw : integrated variance =', M0
    print '  equal dw : estimated Hs        =', 4*M0**0.5

#-------------------------------------------------------------------------------
#
# if outputting coefficients, need to calculate wavenumbers
#
def newton_step(w,k):
    kd = k*d
    tkd = np.tanh(kd)
    return k*tkd - w**2/g, kd*(1-tkd**2) + tkd # F(k), dF/dk
def solve_k(wi,ki):
    niter = 0
    F,J = newton_step(wi,ki)
    if debug: print '  iter',niter,'resid',F
    while niter < MAXITER and np.abs(F) > TOL:
        ki += -F/J
        niter += 1
        F,J = newton_step(wi,ki)
        if debug: print '  iter',niter,'resid',F
    if niter >= MAXITER: print 'WARNING: max newton iterations reached!'
    if debug: 
        if niter < MAXITER: print 'newton iteration converged in',niter,'steps'
    return ki
print 'peak wavenumber (at modal frequency):',solve_k(wm,wm**2/g),'1/m'

k = np.zeros((len(w)))
if write_coeffs:
    k = np.zeros((len(w)))
    kguess = w**2/g
    if verbose: print 'Calculating wavenumbers...'
    for i,wi in enumerate(w):
        #ki = kguess[i]
        #k[i] = ki
        k[i] = solve_k( wi, kguess[i] )

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
dt = Tmax/Nt
print 'Nt=',Nt
t = np.linspace(0,Tmax,Nt)
dt = t[1]-t[0]
freq = fft.fftfreq(Nt,dt) * 2*pi #--fftfreq outputs cycles/time
df = freq[1]-freq[0]
Z = np.zeros((len(t)))
Hs_mean = 0.
A_max_mean = 0.
A_max_abs = 0.
Sfft_mean = np.zeros((Nt))
M0_rand = np.zeros((Nrealizations))
M0_stat = np.zeros((Nrealizations))
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
    if random_amplitude:
        sigma_R = np.sqrt(2/pi) * 2*S*dw #expectation: 2*S*dw
        rand_A2[:] = 0.0
        for i,sig in enumerate(sigma_R):
            if sig==0:
                rand_A2[i] = 0.0
                continue
            rand_A2[i] = np.random.rayleigh(scale=sig)
    else:
        rand_A2 = 2*S*dw
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

    # calculate statistical variance
    exZ = np.mean(Z)
    M0_stat[irand] = np.sum( (Z-exZ)**2 ) / (Nt-1)

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
        t2 = np.linspace(  Tmax,2*Tmax,Nt)
        t3 = np.linspace(2*Tmax,3*Tmax,Nt)
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

    if write_coeffs:
        fname = 'Hs{:.0f}_Tp{:.0f}_{:s}_coeffs{:d}.txt'.format(Hs,Tp,write_name,irand)
        with open(fname,'w') as f:
            f.write('# Irregular wave realization with random amplitude & phase\n')
            f.write('# Discretized with constant delta-w\n')
            f.write('# {:s}\n'.format(time.strftime("%c")))
            f.write('#\n')
            f.write('#       g:\t{:f}\t(m/s^2)\n'.format(g))
            f.write('#       d:\t{:f}\t(m/s^2)\n'.format(d))
            f.write('#      Hs:\t{:f}\t(m)\n'.format(Hs))
            f.write('#      Tp:\t{:f}\t(s)\n'.format(Tp))
            f.write('#      dw:\t{:g}\t(rad/s, bandwidth of each wave component)\n'.format(wmin))
            f.write('#    xref:\t{:f}\t(m, Star wave surface reference location)\n'.format(xref))
            f.write('#  xshift:\t{:f}\t(m, spatial shift applied for Star)\n'.format(xshift))
            f.write('#  tshift:\t{:f}\t(m, temporal shift applied for Star)\n'.format(tshift))
            f.write('#    Tmax:\t{:f}\t(s)\n'.format(Tmax))
            f.write('#      wm:\t{:f}\t(s, modal frequency)\n'.format(wm))
            f.write('#       N:\t{:d}\t(-, number of wave components)\n'.format(Nwavelets))
            f.write('#\n')
            f.write('#\tfrequency\tSpectAmp\tPhase\twavenumber\n')
            f.write('#\t (rad/s) \t (m^2*s)\t(rad)\t   (1/m)  \n')
            # wave elevation, Z(x,t) = sum( S_i**0.5 * cos(k_i*x - w_i*t - phi) ) * dw
            for wi,Si,phii,ki in zip(w,rand_A2/dw**2,-rand_phi,k):
                f.write('{:f}\t{:f}\t{:f}\t{:f}\n'.format( wi, Si, phii, ki ))

    if not debug: pbar.update(irand+1)
if not debug: pbar.finish()
Hs_mean /= Nrealizations
A_max_mean /= Nrealizations
Sfft_mean /= Nrealizations

if verbose:
    print '  absolute max amp.  =', A_max_abs
    print '  average max amp.   =', A_max_mean
    print '  average Hs         =', Hs_mean
    print '  average integrated M0          =', np.mean(M0_rand) # for one-sided spectrum
    print '  variance of integrated M0      =', np.var(M0_rand) # for one-sided spectrum
    print '  estimated Hs(integrated M0)    =', 4*np.mean(M0_rand)**0.5
    print '  average statistical M0         =', np.mean(M0_stat) # for one-sided spectrum
    print '  variance of statistical M0     =', np.var(M0_stat) # for one-sided spectrum
    print '  estimated Hs(statistical M0)   =', 4*np.mean(M0_stat)**0.5

#-------------------------------------------------------------------------------
#
# plot spectrum
#
plotfn = ax.plot #ax.loglog
wplot = np.linspace(wmin,wmax,1000)
plotfn(wplot,wave_spectrum(wplot),'k-',linewidth=2,label='B-S')

plotfn(w,S,'ro',markersize=4,mfc='r',mec='r')

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


if not write_coeffs:
    print '||------------------------||'
    print '||------------------------||'
    print '|| NO OUTPUT WAS WRITTEN! ||'
    print '||------------------------||'
    print '||------------------------||'
plt.show()

