import numpy as np
from scipy import special

def hg(x,y,m,n,w0):
    pm = special.hermite(m)
    pn = special.hermite(n)
    return pm(np.sqrt(2)*x/w0) * pn(np.sqrt(2)*y/w0) * np.exp(-(x**2 + y**2)/w0**2)

def diagonal_hg(x,y,m,n,w0):
    return hg((x-y)/np.sqrt(2),(x+y)/np.sqrt(2),m,n,w0)

def lg(x,y,p,l,w0):
    lag = special.genlaguerre(p,abs(l))
    square_radius = (x**2 + y**2)/w0
    angle = np.arctan2(y,x)
    return (2 * square_radius)**(abs(l)/2) * lag(2 * square_radius) * np.exp(-square_radius + l*angle*1j )

def normalize(holo):
    m = np.amin(holo)
    M = np.amax(holo)
    return np.round((holo - m) / (M- m) * 255).astype('uint8')

def generate_hologram(desired,input,X,Y,L):
    relative = desired / input
    return normalize(np.abs(relative) * (np.mod(np.angle(relative) + 2*np.pi*X / L + np.pi,2*np.pi) - np.pi)) 