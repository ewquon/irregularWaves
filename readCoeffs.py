#!/usr/local/bin/python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
verbose = True

# from input (based on Andy's convention)
x0 = 0.0 # location of device
t0 = 0.0 # time at which the peak should occur (if considering a single focused wave)
tshift = 0.0  # to obtained desired simulation response schedule

xin = -600.0 # inlet location
xout = 600.0 # outlet location
xref = xout # water surface reference location in Star; elevation should be 0 at this point
xshift = 0.0

Tmax = 3*60.*60. # repeat period

update_params_from_file = False

g = 9.801 # m/s^2
d = 70.0 # m

# optional inputs
#init_compare = '/Users/equon/xfer/initsurf.csv'
fname = 'Hs9_Tp15_equalDw_coeffs0.txt'

Nt = 10801
Nx = 501
Tp = 15.1
plot_z_vs_x_at_tinit = True
plot_z_vs_t_at_inlet = True # THIS HAS _NOT_ BEEN CHECKED AS OF 2/25/16
plot_z_vs_t_at_device = False # THIS HAS _NOT_ BEEN CHECKED AS OF 2/25/16
#plot_z_vs_x_over_time = np.arange(0,21)*Tp/5.0

generate_movie_snapshots = False
moviedir = 'TEMP'

est_eta_min = False

# read w, A, p, k from coefficients file {{{
w = [] # frequency, rad/s
S = [] # spectral amplitude, m^2
p = [] # phase, rad
k = [] # wavenumber, rad^2/m

#if not 'fname' in vars():
#    fname = sys.argv[1]
try:
    fname = sys.argv[1]
except IndexError: pass
name = '.'.join(fname.split('.')[:-1])
with open(fname,'r') as f:

    for line in f:
        if line.strip()=='': break
        if line.startswith('#'): 
            sys.stdout.write(line)
            if not update_params_from_file: continue
            # parse additional params...
            line = line.split()
            try:
                param = line[1].split(':')[0]
                if param.startswith('xref'): # time for peak
                    xref = float(line[2])
                    if verbose: print '>>>>>>>>> SETTING',param,'TO',xref,'<<<<<<<<<'
                elif param.startswith('xshift'):
                    xshift = float(line[2])
                    if verbose: print '>>>>>>>>> SETTING',param,'TO',xshift,'<<<<<<<<<'
                elif param.startswith('tshift'):
                    tshift = float(line[2])
                    if verbose: print '>>>>>>>>> SETTING',param,'TO',tshift,'<<<<<<<<<'
                elif param.startswith('Tmax'):
                    Tmax = float(line[2])
                    if verbose: print '>>>>>>>>> SETTING',param,'TO',Tmax,'<<<<<<<<<'
            except IndexError: pass
            continue

        wi,Si,pp,ki = [ float(val) for val in line.split() ]
        w.append(wi)
        S.append(Si)
        p.append(pp)
        k.append(ki)

N = len(w)
w = np.array(w)
try: dw = w[1] - w[0] #dw = np.diff(w)
except IndexError: dw = 1.0
S = np.array(S)
A = np.array(S)**0.5
p = np.array(p)
k = np.array(k)
T = 2*np.pi / w
L = 2*np.pi / k
c = w/k

imax = np.argmax(A)
print '\npeak w={:f} rad/s, T={:f} s, H={:f} m, L={:f} m'.format(w[imax],T[imax],2*A[imax],L[imax])

print N,'modes read'
print 'omega:      [',np.min(w),',',np.max(w),']'
print 'amplitude:  [',np.min(A),',',np.max(A),']'
print 'phase:      [',np.min(p),',',np.max(p),']'
print 'period:     [',np.min(T),',',np.max(T),']'
print 'wavelength: [',np.min(L),',',np.max(L),']'
print 'wavespeed:  [',np.min(c),',',np.max(c),']  at w=',w[np.argmax(c)],'rad/s'

print '==============='

eta_max = np.sum( A*np.cos( k*x0 + p ) ) * dw
print 'max (exact) wave elevation, eta=',eta_max

Asorted = A.copy()
Asorted.sort()
Hs = 2*np.mean( Asorted[-N/3:] )
print 'significant wave height, Hs=',Hs
fS = w*S/(2*np.pi)
M0 = (dw/2/np.pi)*np.sum(fS[:-1]+fS[1:])/2.0
print '           4*sqrt(M0) ~= Hs=',4*np.sqrt(M0)
#print '       for Nmodes = 3*N, Hs=',2*np.mean(A)

print '===============\n\nCHECKS:\n'

i0 = np.nonzero(A)[0][0]
print 'amplitude A > 0 for w >=',w[i0],'rad/s'

disp_check = w**2 - g*k*np.tanh(k*d)
print 'dispersion satisfied? [',np.min(disp_check),',',np.max(disp_check),']'
print 'dispersion satisfied (for w at which A/=0)? [',np.min(disp_check[i0:]),',',np.max(disp_check[i0:]),']'

print 'wavespeed (for w at which A/=0):  [',np.min(c[i0:]),',',np.max(c[i0:]),']  max at w=',w[np.argmax(c)],'rad/s'

#Cg = np.diff(w)/np.diff(k)
#plt.plot(w[:-1]+0.5*np.diff(w),Cg)
#plt.show()
# }}}

#
# set up wave functions
#

def inputsubwave(x,t,phi): return A*np.cos( k*x - w*t - phi )
def Star_subwave(x,t,phi): return A*np.cos( k*(x-xref) - w*t - phi + np.pi/2 )
#def Star_subwave(x,t,phi): return A*np.cos( k*x - w*t - phi - k*(xref-xin) )

# phase correction for Star
centershift = -k*(xref-xshift) + np.pi/2
center_timeshift = centershift + w*tshift

#
# plot
#
if not 'trange' in vars(): trange = np.abs(tshift)

if plot_z_vs_x_at_tinit:# {{{

    x = np.linspace(xin,xout,Nx)
    zinput0 = np.zeros((Nx))
    zinput1 = np.zeros((Nx))
    z_star0 = np.zeros((Nx))
    z_star1 = np.zeros((Nx))
    z_star2 = np.zeros((Nx))
    z_star3 = np.zeros((Nx))
    for i in range(Nx):
        zinput0[i] = np.sum( inputsubwave( x[i]-x0,      t0,         p ) ) * dw
       #zinput1[i] = np.sum( inputsubwave( x[i]-x0,      t0-tshift,  p ) ) * dw
        z_star0[i] = np.sum( Star_subwave( x[i]-x0,      t0,         p ) ) * dw
        z_star1[i] = np.sum( Star_subwave( x[i]-x0,      t0,         p+centershift ) ) * dw
       #z_star2[i] = np.sum( Star_subwave( x[i]-x0,      t0-tshift,  p+centershift ) ) * dw
        z_star3[i] = np.sum( Star_subwave( x[i]-x0,      t0,         p+center_timeshift ) ) * dw
    fig, (ax0,ax1) = plt.subplots(2)
    ax0.plot(x,zinput0,'k-',label='input'.format(t0))
#    ax0.plot(x,zinput0,'b-',label='input (t0={:.1f})'.format(t0))
#    ax0.plot(x,zinput1,'k-',linewidth=2,label='input (time shifted)')
    ax0.set_ylabel('INPUT z(x,t=0)')
    ax0.legend(loc='best')
#    ax1.plot(x,zinput1,'k-',linewidth=2,label='input (time shifted)')
    ax1.plot(x,z_star0,linewidth=2,label='Star')
    ax1.plot(x,z_star1,label='Star (center shift for xout={:.1f})'.format(xout))
#    ax1.plot(x,z_star2,label='Star (center shift, time shift)')
#    ax1.plot(x,z_star3,':',linewidth=4,label='Star (center+time shift)')
    ax1.plot(x,z_star3,label='Star (center+time shift)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('SIMULATED HEIGHT z(x,t=0)')
    titlestr = 'INITIAL profile height'
    try:
        if init_compare:
            xs = []
            zs = []
            with open(init_compare,'r') as f:
                f.readline() #header
                for line in f:
                    #vals = [ float(val) for val in line.split() ]
                    vals = [ float(val) for val in line.split(',') ]
                    xs.append( vals[0] )
                    zs.append( vals[1] )
            ax1.plot(xs,zs,'rs',label='Star (actual)')
            ax1.set_xlim((xin,xout))
            titlestr += ' (Star data: ' + init_compare + ')'
    except NameError: pass
    ax1.legend(loc='best')
    fig.suptitle(titlestr)# }}}

if plot_z_vs_t_at_inlet:# {{{

    t           = np.linspace(0,Tmax,Nt)
    zinput0     = np.zeros((Nt))
    zinput1     = np.zeros((Nt))
    z_star0     = np.zeros((Nt))
    for i in range(Nt):
        zinput0[i] = np.sum( inputsubwave( xin, t[i],        p ) ) * dw
       #zinput1[i] = np.sum( inputsubwave( xin, t[i]-tshift, p ) ) * dw
        z_star0[i] = np.sum( Star_subwave( xin, t[i],        p+centershift ) ) * dw
       #z_star1[i] = np.sum( Star_subwave( xin, t[i],        p+center_timeshift ) ) * dw
    fig, (ax0,ax1) = plt.subplots(2)
    ax0.plot(t,zinput0,'k-',label='input')
    ax0.set_ylabel('INPUT z(x={:.1f},t)'.format(xin))
    ax0.legend(loc='best')
#    ax1.plot(t,zinput1,'k',linewidth=2,label='input (time shifted)')
    ax1.plot(t,z_star0,label='Star')
    ax1.set_xlabel('TIME t')
    ax1.set_ylabel('SIMULATED HEIGHT z(x={:.1f},t)'.format(xin))
    ax1.legend(loc='best')
    fig.suptitle('Trace at INLET')# }}}

if plot_z_vs_t_at_device:# {{{

    t           = np.linspace(0,Tmax,Nt)
    zinput0     = np.zeros((Nt))
    zinput1     = np.zeros((Nt))
    z_star0     = np.zeros((Nt))
    for i in range(Nt):
        zinput0[i] = np.sum( inputsubwave( x0, t[i],        p ) ) * dw
        zinput1[i] = np.sum( inputsubwave( x0, t[i]-tshift, p ) ) * dw
        z_star0[i] = np.sum( Star_subwave( x0, t[i],       -p+center_timeshift ) ) * dw
    fig, (ax0,ax1) = plt.subplots(2)
    ax0.plot(t,zinput0,'k-',label='input')
    ax0.set_ylabel('INPUT z(x={:.1f},t)'.format(x0))
    ax0.legend(loc='best')
    ax1.plot(t,zinput1,'k',linewidth=2,label='input (time shifted)')
    ax1.plot(t,z_star0,label='Star')
    ax1.set_xlabel('TIME t')
    ax1.set_ylabel('SIMULATED HEIGHT z(x={:.1f},t)'.format(x0))
    ax1.legend(loc='best')
    fig.suptitle('Trace at DEVICE')# }}}

if 'plot_z_vs_x_over_time' in vars():# {{{

    x = np.linspace(xin,xout,Nx)
    zinput = np.zeros((Nx))
    zstar  = np.zeros((Nx))
    for i in range(Nx):
        zinput[i] = np.sum( inputsubwave( x[i]-x0, t0, p ) ) * dw
    fig, ax = plt.subplots(1)
    ax.set_xlabel('x')
    ax.set_ylabel('z(x,t)')
    #ax.plot(x,zinput,'k-',linewidth=2,label='input')

    for it,ti in enumerate(plot_z_vs_x_over_time):
        frac = float(it)/(len(plot_z_vs_x_over_time)-1)
        #col = [frac,frac,frac] #[frac,0,1-frac]
        col = 'k'
        for i in range(Nx):
            zstar[i] = np.sum( Star_subwave( x[i]-x0, t0, p+centershift+w*ti ) ) * dw
        ax.plot(x,zstar,color=col,label='t={:f}'.format(ti))
        ax.set_ylim((-6,6))
        fig.suptitle('t={:f}'.format(ti))
        plt.savefig('snapshot{:0>3d}.png'.format(it))
        plt.cla()

    ax.legend(loc='best') # }}}

plt.show()

if est_eta_min or generate_movie_snapshots:# {{{
    x = np.linspace(-600,600,Nx)
    eta_min = 9e9
    t_min = -1
    if generate_movie_snapshots:
        ymax = np.ceil(eta_max)
        plt.figure()
        plt.hold(False)
    print '\nCalculating wave elevations...'
    for t in np.arange(0,Tmax):
        eta = np.zeros(x.shape)
        for i in range(len(x)):
            eta[i] = np.sum( A*np.cos( k*x[i] - w*t + p ) ) * dw
        if generate_movie_snapshots:
            plt.plot(x, eta)
            plt.ylim((-ymax,ymax))
            plt.title('t=%f'%t)
            imgname = moviedir + os.sep + '{:s}_{:04d}.png'.format(name,int(t))
            plt.savefig(imgname)
            print 'wrote',imgname
        else:
            curval = np.min(eta)
            if curval < eta_min:
                eta_min = curval
                t_min = t
    print 'min (estimated) wave elevation eta:',eta_min,'at',t_min,'s'
# }}}

