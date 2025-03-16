import numpy as np
import matplotlib.pyplot as plt
from newton import newton
    
def f(x,points):
        h, k, r = x
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        f1 = (x1 - h)**2 + (y1 - k)**2 - r**2
        f2 = (x2 - h)**2 + (y2 - k)**2 - r**2
        f3 = (x3 - h)**2 + (y3 - k)**2 - r**2
        return np.array([f1, f2, f3])

def J(x, points):
        h, k, r = x
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        return np.array([
            [-2*(x1 - h), -2*(y1 - k), -2*r],
            [-2*(x2 - h), -2*(y2 - k), -2*r],
            [-2*(x3 - h), -2*(y3 - k), -2*r]
        ])

donnée = {
    "A": [(1, 1), (0, 0.73205), (-0.73205, 0)],
    "B": [(-1, 0), (-1.0066, 0.1147), (-1.136, 0.50349)],
    "C": [(0, 1), (-0.0112, 1.149247), (-0.0465, 1.301393)] }

x0 = np.array([0, 0, 1])
tolr = 10**-10
nmax = 100
np.set_printoptions(precision=16)

for key, item in donnée.items():
    newton(f(x0, item), J(x0, item), x0, tolr, nmax)