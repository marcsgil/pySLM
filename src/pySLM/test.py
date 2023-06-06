import numpy as np
import slmpy
import initial_profiles

#All the distances are in mm
#The SLM parameters are adjusted for a Hamamatsu X10468
width = 15.8
height = 12

slm = slmpy.SLMdisplay(monitor = 0)
resX, resY = slm.getSize()
X,Y = np.meshgrid(np.linspace(-width/2 , width/2 ,resX),np.linspace(-height/2 ,height/2 ,resY))
xoffset = 0.2
yoffset = .05
input = initial_profiles.hg(X-xoffset,Y-yoffset,0,0,2)
desired = initial_profiles.diagonal_hg(X-xoffset,Y-yoffset,1,1,.25)

holo = initial_profiles.generate_hologram(desired,input,X,Y,.1)

slm.updateArray(holo) #Mais um
slm.close()

#Comentário do altilano

