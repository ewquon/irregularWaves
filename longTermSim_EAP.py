#!/usr/local/bin/python
import numpy as np
import matplotlib.pyplot as plt
import peaks # for estimating Hs
from scipy import optimize # for calculating panel edge positions
from scipy import integrate
from numpy import fft
from progressbar import ProgressBar, Bar, Percentage
np.seterr(divide='ignore')
pi = np.pi
twopi = 2.0*pi

#############################################
# INPUTS
verbose = True
debug = False
g  = 9.81
d  = 70.0           # water depth, m
Tp = 15.1           # peak period, s
Hs = 9.0            # significant wave height, m
wm = twopi/Tp       # modal frequency, rad/s
wmin = wm/2         # smallest frequency
wmax = 10*wm        # largest frequency

def wave_spectrum(w):    # Bretschneider
    wnorm4 = (w/wm)**4
    return 1.25/4.0 / (w*wnorm4) * Hs**2 * np.exp(-1.25/wnorm4) #m^2/(rad/s)

Nwavelets = 10
Nrealizations = 1 #500
plot_realizations = False
Nt = 500 # number of points in each realization

TOL = 1e-14
MAXITER = 10
#############################################

Tmax = Nwavelets*twopi/(wmax-wmin) # time after which the wave train is repeated

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
M0 = integrate.romberg(wave_spectrum, wmin, wmax, show=verbose, tol=TOL)
panelarea = M0 / Nwavelets
#intSdw_quad, abserr, info = integrate.quad(wave_spectrum, wmin, wmax, full_output=True)
#print intSdw_quad

# calculate panel edge locations
w0 = np.zeros((Nwavelets+1))
w0[0] = wmin
B = 1.25*wm**4
for i in range(Nwavelets):
    w0[i+1] = ( -B / np.log( 16/Hs**2*panelarea + np.exp(-B/w0[i]**4) ) )**0.25
print 'wmax err:',wmax-w0[-1]
w0[-1] = wmax

# optimize positions to get ~equal panel area
#plt.figure()
#w0 = np.linspace(wmin,wmax,Nwavelets+1) # frequencies at panel edges, initially evenly distributed
#plt.plot(w0,np.zeros((Nwavelets+1)),'k+')

#wi = w0[1:-1]
#def delta_areas(wi):
#    w = w0.copy()
#    w[1:-1] = wi
#    wmid = (w[1:] + w[:-1]) / 2.0
#    print 'wmid',wmid
#    Smid = wave_spectrum( wmid )
#    dw = np.diff(w)
#    print Smid*dw
#    return Smid*dw - panelarea
##    wi_m1 = w[:-2]
##    wi_p1 = w[2:]
##    Si    = wave_spectrum( (wi + wi_m1)/2.0 )
##    Si_p1 = wave_spectrum( (wi + wi_p1)/2.0 )
##    print Si
##    print Si_p1
##    return (wi - wi_m1)*Si - (wi_p1 - wi)*Si_p1
#print 'delta areas (before)',delta_areas(wi)
#result = optimize.leastsq( delta_areas, wi )
#wnew = result[0]
#print 'delta areas (after)',delta_areas(wnew)
#for before,after in zip(w0[1:-1],wnew):
#    print before,after
#w0[1:-1] = wnew

#def delta_area_norm(w0):
#    wmid = (w0[1:] + w0[:-1]) / 2.0
#    Smid = wave_spectrum( wmid )
#    delta_areas = Smid*np.diff(w0) - panelarea
#    return delta_areas.dot( delta_areas )
#print delta_area_norm(w0)
#bnds = [ (wmin,wmax) for i in range(Nwavelets+1) ]
#cons = ({'type':'eq', 'fun': lambda x: x[0] - wmin},
#        {'type':'eq', 'fun': lambda x: x[-1] - wmax},
#        {'type':'ineq', 'fun': lambda x: x[1:] - x[:-1]})
#result = optimize.minimize( delta_area_norm, w0, method='SLSQP', bounds=bnds, constraints=cons )
#print result.message
#for before,after in zip(w0,result.x):
#    print before,after
#w0 = result.x
#print w0
#print delta_area_norm(w0)

#wnew = w0.copy()
#for i in range(1,Nwavelets):
#    fn = lambda x: (x-w0[i-1])*wave_spectrum((x+w0[i-1])/2.0) - panelarea
#    wnew[i] = optimize.fsolve( fn, wnew[i-1] )
#for before,after in zip(w0,wnew):
#    print before,after
#w0 = wnew

dw = np.diff(w0)
w = (w0[1:] + w0[:-1]) / 2.0
S = wave_spectrum( w )
#plt.plot(w,np.zeros((Nwavelets)),'r*')
print panelarea,S*dw
plt.show()

if verbose: 
    print 'Note that differences are due to wmin > 0 and wmax < inf'
    print 'exact: estimated Hs from spectrum    =',4*M0**0.5
    print ''
    print 'equal dw: calculated variance        =',M0_equal
    print 'equal dw: estimated Hs from spectrum =',4*M0_equal**0.5
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
        rand_A2[i] = np.random.rayleigh(scale=sig)
    rand_A = rand_A2**0.5
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

if verbose:
    print '  average Hs         =',Hs_mean
    print '  average M0         =',np.mean(M0_rand) # for one-sided spectrum
    print '  variance of M0     =',np.var(M0_rand) # for one-sided spectrum
    print '  estimated Hs(M0)   =',4*np.mean(M0_rand)**0.5

#-------------------------------------------------------------------------------
#
# plot spectrum
#
wplot = np.linspace(wmin,wmax,1000)
ax.plot(wplot,wave_spectrum(wplot),'k-',linewidth=2,label='B-S')
ax.plot(w,S,'ks')
for wi in w0: ax.plot([wi,wi],[0,wave_spectrum(wi)],'0.8')
ax.set_ylabel('S [m^2/(rad/s)]')
# TODO check: since we threw out the negative-frequency part of spectrum, we lost half the power in the spectrum...
#             thus the factor of two?
ax.plot(freq[:Nt/2],2*Sfft_mean[:Nt/2],label='avg over {:d} realizations (equal dw)'.format(Nrealizations))
ax.set_xlim((0,wmax))
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

