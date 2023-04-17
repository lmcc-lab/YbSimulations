# Atomic Checklist
Use Yb174. In the order I would do them on a given day
- **RF**
- **Vaccum**
- **Voltages well**
- **laser frequency** iF THERES A PROBLEM MAYBE TRY DETUNING FURTHER
- **laser power**
- **Saturation power**
- **Beam weist**
- **Rayleigh Range**

- **Check beams overlap**. Use beam profiler in two different positions to check beam overlap.
- **Make sure beams are going through center of lens**. Set the shuttle position to the mirror position (864) and check the lasers are going through the center of the lens.
- **Check cooling laser polarisation**. Should be horizontal with B field (For 174 only).
- **Check resonance**. Sweep 935 frequency and fit Gaussian to check where the resonance is. Can also use 370 but make sure you don't go above 0 Mhz detuning or else you could loose the ion as it heats up. 
- **935 polarisation**. Move fiber paddles to maximise counts, then move HWP to also maximise counts. 
- **D state pump times**. Measuring the pump times into and out of the $D$ state can tell you how well things are hitting the ion independent of anything else.
- **Compensating for external mangetic fields**. Henle resonance; turn off main coil (z axis), then minimize counts on all three compensation coils independently. Counts should drop considerably (in the thousands). Turn back on main coil, counts should go back up to where it was if not higher.
- **935 saturation**. Turn down 935 power and measure counts. It should be set to saturate so a small change in power doesn't change counts.
- **Check beam pointing**. Move lens position to maximise counts/doppler saturation parameter.
- **Check beam spot size on ion**. Move lens out of focus, recording how far it's been moved. Then move x-y to get an idea of the spot size. Then begin moving back into focus and past 
- **B field alignment using Raman beams**. If the polarisation of the Raman beams is set correctly to $\sigma_-$, $\sigma_+$ with the correct detunings, then when you have both Raman beams hitting the ion it should fluoresce. When only one is on then it should go completely dark. 
- **Run Electrode Compensation code**. Compensate for stray electric fields.
- **370 power saturation**. Measure the saturation power of cooling laser. Use lens position to maximise it.  
- **Check zeeman splitting**. Increasing zeeman shift generally reduces counts with Yb174.
- **Adjust weights of electrodes**. Ex, Ey will move the ion with the electrodes and can sometimes help increase counts by position the ion better in the path of the lasers.


Once these have all been done then the counts 'should' be higher, ideally about 160,000 c/s.