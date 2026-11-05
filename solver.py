import numpy as np


def schrodinger_exp(a, b, m, n, T, V, psi0):
    # Définition de l'espace
    x = np.linspace(a, b, m + 1)
    dx = x[1] - x[0]

    # Définition du temps
    t = np.linspace(0, T, n + 1)
    dt = t[1] - t[0]

    # On initialise psi à 0. Psi est tq psi[i][j] contient la valeur psi(x_i, t_j)
    psi = np.zeros((m + 1, n + 1), dtype=complex)

    # psi(x, 0) = psi0 la condition initiale
    psi[: , 0] = psi0(x)

    # Conditions aux limites pour x in [a,b] : psi(a,t) = psi(b,t) = 0
    psi[0, :] = 0
    psi[-1, :] = 0

    Vx = V(x[1:m]) if callable(V) else V

    for k in range(n):
        psi[1:m, k+1] = psi[1:m, k] + 1j*dt/(2*dx*dx)*(psi[:-2,k] - 2*psi[1:-1,k] + psi[2:,k]) - 1j*dt*Vx*psi[1:m,k]

    return x, t, psi


def schrodinger_semi_imp(a, b, m, n, T, V, psi0):
    # Définition de l'espace
    x = np.linspace(a, b, m + 1)
    dx = x[1] - x[0]

    # Définition du temps
    t = np.linspace(0, T, n + 1)
    dt = t[1] - t[0]

    # On initialise psi à 0. Psi est tq psi[i][j] contient la valeur psi(x_i, t_j)
    psi = np.zeros((m + 1, n + 1), dtype=complex)

    # psi(x, 0) = psi0 la condition initiale
    psi[:, 0] = psi0(x)

    # Conditions aux limites pour x in [a,b] : psi(a,t) = psi(b,t) = 0
    psi[0, :] = 0
    psi[-1, :] = 0

    # Soit V est une fonction soit un scalaire
    Vx = V(x[1:m]) if callable(V) else V

    # Construction des matrices A et B

    A = np.diagflat(np.full(m-2, 1/(4*dx*dx)), -1) + np.diagflat(np.full(m-2, 1/(4*dx*dx)), 1) + np.diagflat(np.full(m-1, 1j/dt - 1/(2*dx*dx) - Vx/2), 0)
    B = -np.conj(A)

    for k in range(n):
        psi[1:m, k+1] = np.linalg.solve(A, B @ psi[1:m, k])

    return x, t, psi
