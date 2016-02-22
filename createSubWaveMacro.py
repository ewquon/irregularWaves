#!/usr/bin/python
import sys
from math import pi

if len(sys.argv) <= 1:
    print 'USAGE:',sys.argv[0],'coeffs.txt'
    sys.exit()

t_peak = 0.0 # time offset
x_start = 0.0 # inflow location relative to device
output_zero_amplitude = False
verbose = True

# TODO: Update header as needed (e.g. to change the name of the physics
#       continuum or the superposition VOF wave
headerstr = """// STAR-CCM+ macro: addSubWave.java
package macro;

import java.util.*;

import star.common.*;
import star.base.neo.*;
import star.vof.*;

public class {macroName:s} extends StarMacro {{

  public void execute() {{
    execute0();
  }}

  private void execute0() {{

    Simulation simulation_0 = getActiveSimulation();
    PhysicsContinuum physicsContinuum_0 = ((PhysicsContinuum) simulation_0.getContinuumManager().getContinuum("Physics 1"));
    VofWaveModel vofWaveModel_0 = physicsContinuum_0.getModelManager().getModel(VofWaveModel.class);

    SuperpositionVofWave supWave = ((SuperpositionVofWave) vofWaveModel_0.getVofWaveManager().getObject("SuperpositionVofWave 1"));
    
    //*** WAVE COMPONENTS INPUT BELOW***
"""

# TODO: Update subwave specification as needed
wavestr = """
    FirstOrderSuperposingVofWave subwave{waveIdx:d} = 
      supWave.getSuperposingVofWaveManager().createSuperposingVofWave(FirstOrderSuperposingVofWave.class, "FirstOrderSuperposing");
    subwave{waveIdx:d}.getAmplitude().setValue({:f}); // m
    subwave{waveIdx:d}.getPhase().setValue({:f}); // radians
    subwave{waveIdx:d}.getSpecificationOption().setSelected(VofWaveSpecificationOption.WAVE_PERIOD_SPECIFIED);
    ((VofWavePeriodSpecification) subwave{waveIdx:d}.getVofWaveSpecification()).getWavePeriod().setValue({:f}); // seconds
"""

closestr = """
  }
}"""

#
# EXECUTION STARTS HERE
#

fname = sys.argv[1]
name = '.'.join(fname.split('.')[:-1])

#-- first time: get delta omega
dw = 0
with open(fname,'r') as f:
    for line in f:
        if line.startswith('#'): continue
        line = line.split()
        if dw==0: 
            dw = -float(line[0])
        elif dw < 0: 
            dw += float(line[0])
            break
        else:
            print 'shouldn''t be here!!!'
            break
if verbose: print 'dw=',dw

#-- second time: process each mode and write to file or stdout
out = open(name+'.java','w')
with open(fname,'r') as f:

    if out: out.write(headerstr.format(macroName=name))
    else: print headerstr.format(macroName=name)

    iwave = 0
    for line in f:
        if line.strip()=='': break
        if line.startswith('#'): 
            if verbose: sys.stdout.write(line)
            # parse additional params...
            line = line.split()
            try:
                param = line[1]
                if param.startswith('t0'): # time for peak
                    t0 = float(line[2])
                    if verbose: print '>>>>>>>>> SETTING',param,'TO',t0,'<<<<<<<<<'
                elif param.startswith('startTime'): # time for peak
                    t_start = float(line[2])
                    if verbose: print '>>>>>>>>> SETTING',param,'TO',t_start,'<<<<<<<<<'
                elif param.startswith('t_peak'): # time at start of simulation
                    t_peak = float(line[2])
                    if verbose: print '>>>>>>>>> SETTING',param,'TO',t_peak,'<<<<<<<<<'
                elif param.startswith('x_start'):
                    x_start = float(line[2])
                    if verbose: print '>>>>>>>>> SETTING',param,'TO',x_start,'<<<<<<<<<'
            except IndexError: pass
            continue

        # At this point we should be done reading the header...
        # perform sanity checks:
        try:
            assert( t_peak >= 0 )
        except NameError:
            t_peak = t0 - t_start
            print 'Calculated time at peak response =',t_peak

        # z(x,t) = dw * np.sum( A*np.cos( k*x - w*(t-toffset) + phi ) )
        #A, phi, T = [ float(val) for val in line.split() ]
        #print A, phi, T
        w, S, phi, k = [ float(val) for val in line.split() ]
        if not output_zero_amplitude and S==0: continue
        T = 2*pi/w
        A = dw * S**0.5

        # adjustments for domain size
        # w/o correction, the peak should occur at x=0, t=0
        #TODO: currently assuming that -x_start == xout
        #shift = -k*xout + pi/2 - w*t_peak
        shift = k*x_start - w*t_peak + pi/2

        # A: [m], phi: [rad], T: [s]
# changed 2/10/16
# new formulation has different sign on phi
#        if out: out.write(wavestr.format(A, -phi+shift, T, waveIdx=iwave))
#        else: print wavestr.format(A, phi, T, waveIdx=iwave)
        if out: out.write(wavestr.format(A, phi+shift, T, waveIdx=iwave))
        else: print wavestr.format(A, phi+shift, T, waveIdx=iwave)

        iwave += 1

    if out: out.write(closestr)
    else: print closestr

out.close()
print 'wrote',name+'.java'
