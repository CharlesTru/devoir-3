import numpy as np
import matplotlib.pyplot as plt
from newton import newton

def f(x, points):
    h, k, r = x
    return np.array([
        (points[0][0] - h)**2 + (points[0][1] - k)**2 - r**2,
        (points[1][0] - h)**2 + (points[1][1] - k)**2 - r**2,
        (points[2][0] - h)**2 + (points[2][1] - k)**2 - r**2
    ])


def J(x, points):
    h, k, r = x
    return np.array([
        [-2 * (points[0][0] - h), -2 * (points[0][1] - k), -2 * r],
        [-2 * (points[1][0] - h), -2 * (points[1][1] - k), -2 * r],
        [-2 * (points[2][0] - h), -2 * (points[2][1] - k), -2 * r]])

donnée = {
    "A": [(1, 1), (0, 0.73205), (-0.73205, 0)],
    "B": [(-1, 0), (-1.0066, 0.1147), (-1.136, 0.50349)],
    "C": [(0, 1), (-0.0112, 1.149247), (-0.0465, 1.301393)] }

x0 = np.array([0, 0, 1])
tolr = 10**-10
nmax = 100
np.set_printoptions(precision=16)

for key, points in donnée.items():
    (x1, y1), (x2, y2), (x3, y3) = points
    solution = newton(f(x0, points), J(x0, points), x0, tolr, nmax)

print(solution)