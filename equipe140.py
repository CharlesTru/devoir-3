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

En_A = np.linalg.norm(SolA[:,1:] - SolA[:,:-1], axis=0)
En_B = np.linalg.norm(SolB[:,1:] - SolB[:,:-1], axis=0)
En_C = np.linalg.norm(SolC[:,1:] - SolC[:,:-1], axis=0)

plt.figure(1)
plt.semilogy(En_A, label='En_A = ∥⃗xn+1 − ⃗xn∥')
plt.semilogy(En_B, label='En_B = ∥⃗xn+1 − ⃗xn∥')
plt.semilogy(En_C, label='En_C = ∥⃗xn+1 − ⃗xn∥')
plt.xlabel("Itération n")
plt.ylabel("Erreur |x_n+1 - x_n|")
plt.title("Évolution de l'erreur en fonction du nombre d'itération en semi-log")
plt.legend()

A = np.array([(1, 1), (0, 0.73205), (-0.73205, 0)])
B = np.array([(-1, 0), (-1.0066, 0.1147), (-1.136, 0.50349)])
C = np.array([(0, 1), (-0.0112, 1.149247), (-0.0465, 1.301393)])


hA, kA, rA = SolA[:,-1]
hB, kB, rB = SolB[:,-1]
hC, kC, rC = SolC[:,-1]

fig, ax = plt.subplots(figsize=(8, 8)) 
ax.set_aspect('equal') 

ax.scatter(A[:, 0], A[:, 1], label="Données A", color='blue', marker='o')
ax.scatter(B[:, 0], B[:, 1], label="Données B", color='red', marker='s')
ax.scatter(C[:, 0], C[:, 1], label="Données C", color='green', marker='d')

cercle_A = plt.Circle((hA, kA), rA, color='blue', fill=False, linestyle='dashed', linewidth=1.5, label="Cercle A")
cercle_B = plt.Circle((hB, kB), rB, color='red', fill=False, linestyle='dashed', linewidth=1.5, label="Cercle B")
cercle_C = plt.Circle((hC, kC), rC, color='green', fill=False, linestyle='dashed', linewidth=1.5, label="Cercle C")

ax.add_patch(cercle_A)
ax.add_patch(cercle_B)
ax.add_patch(cercle_C)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Cercle passant par les points des jeux A, B et C")
ax.legend()
ax.grid(True, linestyle="--", linewidth=0.5)

#QUESTION 2
def vandermonde_modifiee(Q):
    n = len(Q) - 1
    V = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(n+1):
            V[i, j] = Q[i]**(n-j)
    return V

# Partie (a)
n_values = np.arange(10, 151, 10)
cond_values = []

for n in n_values:
    Qn = np.array([-1 + 2*i/n for i in range(n+1)])
    V = vandermonde_modifiee(Qn)
    cond = np.linalg.cond(V, 'fro')
    cond_values.append(cond)

# Figure 3
plt.figure(figsize=(10, 6))
plt.semilogy(n_values, cond_values, 'b.-', label='cond(Ṽ(Qn))')
plt.semilogy(n_values, 2**n_values, 'r--', label='2^n')
plt.grid(True)
plt.xlabel('n')
plt.ylabel('Conditionnement (échelle log)')
plt.title('Conditionnement de Ṽ(Qn) vs 2^n')
plt.legend()
plt.show()

# Partie (b)
log_cond = np.log(cond_values)
coeff = np.polyfit(n_values, log_cond, 1)
a = coeff[0]
b = np.exp(a)

print(f"Valeur de b estimée : {b}")

# Figure 4
n_range = np.arange(1, 101)
plt.figure(figsize=(10, 6))
plt.semilogy(n_range, b**n_range, 'b-', label=f'b^n (b={b:.3f})')
plt.semilogy(n_range, 2**n_range, 'r--', label='2^n')
plt.grid(True)
plt.xlabel('n')
plt.ylabel('Valeur (échelle log)')
plt.title('Comparaison de b^n et 2^n')
plt.legend()
plt.show()

# Partie (c)
n_values_c = np.arange(30, 201)
errors = []
eps_machine = np.finfo(np.float64).eps

for n in n_values_c:
    Qn = np.array([-1 + 2*i/n for i in range(n+1)])
    V = vandermonde_modifiee(Qn)
    b_vec = np.ones(n+1)
    x_true = np.zeros(n+1); x_true[-1] = 1
    x_star = np.linalg.solve(V, b_vec)
    error = np.linalg.norm(x_true - x_star, 2) / np.linalg.norm(x_true, 2)
    errors.append(error)

# Figure 5
plt.figure(figsize=(10, 6))
plt.semilogy(n_values_c, errors, 'b.-', label='Erreur relative')
plt.semilogy(n_values_c, eps_machine * b**n_values_c, 'r--', 
             label=f'ε_m * b^n (b={b:.3f})')
plt.grid(True)
plt.xlabel('n')
plt.ylabel('Erreur (échelle log)')
plt.title('Erreur relative et borne théorique')
plt.legend()
plt.show()