#!/usr/local/bin/python
import sys
import numpy as np
np.seterr(divide='ignore')
pi = np.pi
twopi = 2.0*pi

#############################################
# INPUTS
verbose = False
TOL = 1e-14

g  = 9.81
d  = 70.0           # water depth, m
Tp = 15.1           # peak period, s
Hs = 9.0            # significant wave height, m
wm = twopi/Tp       # modal frequency, rad/s
#wmin = wm/2         # smallest frequency, rad/s
wmax = 50*wm        # largest frequency, rad/s

#Nwavelets = 100 # ~8000 needed to get Tmax=3 hrs with equal dw, wmax=10*wm
Nwavelets = int(sys.argv[1])
def wave_spectrum(w):    # Bretschneider
    wnorm4 = (w/wm)**4
    return 1.25/4.0 / (w*wnorm4) * Hs**2 * np.exp(-1.25/wnorm4) #m^2/(rad/s)

#############################################

wmin = wmax/Nwavelets

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

dw = np.diff(w0)
w = (w0[1:] + w0[:-1]) / 2.0
S = wave_spectrum( w )
M0 = S.dot( dw )

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

print Nwavelets, M0_equal,4*M0_equal**0.5, M0,4*M0**0.5

