import numpy as np
import matplotlib.pyplot as plt

def f(h, k, r):
    h, k, r = x
    return np.array([
        (x1 - h)**2 + (y1 - k)**2 - r**2,
        (x2 - h)**2 + (y2 - k)**2 - r**2,
        (x3 - h)**2 + (y3 - k)**2 - r**2
    ])

def J(x):
    h, k, r = x
    return np.array([
        [-2*(x1 - h), -2*(y1 - k), -2*r],
        [-2*(x2 - h), -2*(y2 - k), -2*r],
        [-2*(x3 - h), -2*(y3 - k), -2*r]
    ])

donnée = {
    "A": [(1, 1), (0, 0.73205), (-0.73205, 0)],
    "B": [(-1, 0), (-1.0066, 0.1147), (-1.136, 0.50349)],
    "C": [(0, 1), (-0.0112, 1.149247), (-0.0465, 1.301393)] }

x0 = np.array([0.0, 0.0, 1.0])
tolr = 1e-10
nmax = 100
np.set_printoptions(precision=16)