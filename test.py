#!/usr/local/bin/python
import numpy as np
import matplotlib.pyplot as plt
import waves

# inputs
Tp = 15.1
Tmax = 2*Tp
phi = 0. #np.pi/2
phi2= 0.5

xin = -600.0
xout = 600.0
Nx = 1201
Nt = 501

xref = xout

# calculations
w = 2*np.pi / Tp
k = waves.solve_k(w)
L = 2*np.pi / k
print 'phase:',phi,'rad'
print 'Tp=',Tp,'s'
print 'w =',w,'rad/s'
print 'k =',k,'1/m'
print 'L =',L,'m'

w2 = w/2
k2 = waves.solve_k(w2)
print 'L(2)=',2*np.pi/k2

# plot
def inputsubwave(x,t,k,w,phi): return np.cos( k*x - w*t - phi )
def Star_subwave(x,t,k,w,phi): return np.cos( k*(x-xref) - w*t - phi + np.pi/2 )

x = np.linspace(xin,xout,Nx)
t = np.linspace(0,Tmax,Nt)

sample = Star_subwave(x,0.0,k,w,phi) + 0.5*Star_subwave(x,0.0,k2,w2,phi2)
print sample[-25:]

plt.figure()
plt.plot( x, inputsubwave(x,0.0,k,w,phi), label='input' )
plt.plot( x, Star_subwave(x,0.0,k,w,phi), label='Star', linewidth=2 )
#plt.plot( x, inputsubwave(x,0.0,k,w,phi) + 0.5*inputsubwave(x,0.0,k2,w2,phi2), label='input' )
#plt.plot( x, Star_subwave(x,0.0,k,w,phi) + 0.5*Star_subwave(x,0.0,k2,w2,phi2), label='Star', linewidth=2 )
plt.title( 'profile at t=0' )
plt.legend()

#plt.figure()
#plt.plot( t, inputsubwave(0.0,t,k,w,phi), label='input' )
#plt.plot( t, Star_subwave(0.0,t,k,w,phi), label='Star' )
#plt.title( 'trace at x=0' )
#plt.legend()



#-----------------------
plt.show()
