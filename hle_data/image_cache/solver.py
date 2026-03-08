import numpy as np

dx = 0.1
dy = 0.1
lmbda = (dx/dy)**2
alpha = lmbda + 1

def f(i, j):
    x = i * dx
    y = j * dy
    return 100 * x * y

# Node mapping from analysis:
# T1: (1, 2)
# T2: (2, 2)
# T3: (1, 1)
# T4: (2, 1)
nodes = [(1, 2), (2, 2), (1, 1), (2, 1)]
node_to_idx = {pos: i for i, pos in enumerate(nodes)}

A = np.zeros((4, 4))
b = np.zeros(4)

for idx, (i, j) in enumerate(nodes):
    A[idx, idx] = 2 * alpha
    rhs = -(dx**2) * f(i, j)
    
    # Neighbors: (i, j+1), (i, j-1), (i+1, j), (i-1, j)
    neighbors = [(i, j+1), (i, j-1), (i+1, j), (i-1, j)]
    for k, (ni, nj) in enumerate(neighbors):
        coeff = lmbda if k < 2 else 1
        if (ni, nj) in node_to_idx:
            A[idx, node_to_idx[(ni, nj)]] -= coeff
        else:
            # Boundary conditions: T(0,y)=0, T(0.3,y)=1, T(x,0)=0, T(x,0.3)=0.5
            val = 0
            if ni == 0: val = 0
            elif ni == 3: val = 1.0
            elif nj == 0: val = 0
            elif nj == 3: val = 0.5
            rhs += coeff * val
    b[idx] = rhs

T = np.linalg.solve(A, b)
print(list(T))
