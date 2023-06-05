import numpy as np
import slmpy
import initial_profiles

slm = slmpy.SLMdisplay(monitor=0)
resX, resY = slm.getSize()
delta = 3 / resY
X,Y = np.meshgrid(np.linspace(-delta * resX,delta * resX,resX),np.linspace(-3,3,resY))
input = initial_profiles.hg(X,Y,0,0,5)
desired = initial_profiles.lg(X,Y,1,2,.1)

holo = initial_profiles.generate_hologram(desired,input,X,Y,.1)

slm.updateArray(holo)
slm.close()