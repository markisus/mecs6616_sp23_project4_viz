import numpy as np

def rot(theta):
    R = np.zeros((2,2))
    R[0,0] = np.cos(theta)
    R[0,1] = -np.sin(theta)
    R[1,0] = np.sin(theta)
    R[1,1] = np.cos(theta)
    return R

def xaxis():
    x = np.zeros((2, 1))
    x[0] = 1
    return x

def yaxis():
    y = np.zeros((2, 1))
    y[1] = 1
    return y

def wrap(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi