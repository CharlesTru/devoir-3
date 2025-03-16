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


A = [(1, 1), (0, 0.73205), (-0.73205, 0)]
B = [(-1, 0), (-1.0066, 0.1147), (-1.136, 0.50349)]
C = [(0, 1), (-0.0112, 1.149247), (-0.0465, 1.301393)]

x0 = np.array([0, 0, 1])
tolr = 10**-10
nmax = 100
np.set_printoptions(precision=16)

SolA = newton(lambda x: f(x, A), lambda x: J(x,A), x0, tolr, nmax)
SolB = newton(lambda x: f(x, B), lambda x: J(x,B), x0, tolr, nmax)
SolC = newton(lambda x: f(x, C), lambda x: J(x,C), x0, tolr, nmax)

lastiterA = SolA[:,-1]
lastiterB = SolB[:,-1]
lastiterC = SolC[:,-1]

print(lastiterA)
print(lastiterB)
print(lastiterC)
detA = np.linalg.det(J(lastiterA, A))
detB = np.linalg.det(J(lastiterB, B))
detC = np.linalg.det(J(lastiterC, C))

print(detA)
print(detB)
print(detC)