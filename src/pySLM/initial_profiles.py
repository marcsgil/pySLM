import numpy as np
from scipy import special

def hg(x, y, n, m, w0, **kwargs):
    pm = special.hermite(m)
    pn = special.hermite(n)
    var = {
        'lamb' : 1064,
        'z' : 0,
    }
    var.update(kwargs)
    zr = np.pi*w0**2/var['lamb']
    w = w0*np.sqrt(1 + (var['z']/zr)**2)
    gouy = -1j*(n + m + 1)*np.arctan(var['z']/zr)
    E = (pn(np.sqrt(2)*x/w)*pm(np.sqrt(2)*y/w) * 
        np.exp(-(1-1j*var['z']/zr)*(x**2 + y**2)/w**2 + gouy))
    return E

def diagonal_hg(x,y,m,n,w0, **kwargs):
    return hg((x-y)/np.sqrt(2),(x+y)/np.sqrt(2),m,n,w0, **kwargs)

def lg(x, y, p, l, w0, **kwargs):
    var = {
        'z' : 0,
        'lamb' : 1064,
    }
    var.update(kwargs)
    lag = special.genlaguerre(p,abs(l))
    zr = np.pi*w0**2/var['lamb']
    k = 2*np.pi/var['lamb']
    w = w0*np.sqrt(1 + (var['z']/zr)**2)
    R = (var['z']**2 + zr**2)
    gouy = (np.abs(l) + 2*p + 1)*np.arctan(var['z']/zr)
    r = np.sqrt((x**2 + y**2)/w)
    E = ((np.sqrt(2)*r)**(np.abs(l)) * np.exp(-(r)**2) * lag(2*(r)**2) * np.exp(-1j*(k*var['z']*(x**2 + y**2)/(2*R) + l*np.arctan(y/x) - gouy)))
    return E

def normalize(holo):
    m = np.amin(holo)
    M = np.amax(holo)
    return np.round((holo - m) / (M- m) * 255).astype('uint8')

def generate_hologram(desired,input,X,Y,L):
    relative = desired / input
    return normalize(np.abs(relative) * (np.mod(np.angle(relative) + 2*np.pi*X / L + np.pi,2*np.pi) - np.pi)) 

def lens(x, y, fx, fy, lamb):
    k = 2*np.pi/lamb
    return np.exp(-1j*k*((x**2)/fx + (y**2)/fy))

def tilted_lens(x, y, f, theta, lamb):
    fx = f*np.cos(theta)
    fy = f/np.cos(theta)
    return lens(x, y, fx, fy, lamb)