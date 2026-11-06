import numpy as np
from solver import *
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


''' Simulation A : Le puits infini '''

V = lambda x: np.zeros_like(x)
psi0 = lambda x: np.sqrt(30) * x * (1 - x)

x_imp, t_imp, psi_imp = schrodinger_semi_imp(0,1,100,200,1.3,V, psi0)

# Densité de probabilité
rho = np.abs(psi_imp)**2

plt.plot(x_imp, rho)
plt.show()


# Si on veut voire la corde de guitare : rho = np.real(psi_imp) et ax.set_ylim(-2, 2)
# Explicite : x_exp, t_exp, psi_exp = schrodinger_exp(0,1,100,20000,1.3,V, psi0)
# plt.plot(x_exp, np.abs(psi_exp)**2)


# Animation
fig, ax = plt.subplots(figsize=(8, 5))
ligne, = ax.plot(x_imp, rho[:, 0] , 'b-', lw=2)

# Limites et légendes du graphe
ax.set_xlim(0, 1)
ax.set_ylim(0, np.max(rho) * 1.1)
ax.set_xlabel("x")
ax.set_ylabel(r"$|\psi(x,t)|^2$")
ax.grid(True)

# Mise à jour de chaque frame
def update(frame):
    ligne.set_ydata(rho[:, frame])
    ax.set_title(f"Puits infini : t = {t_imp[frame]:.2f}")
    return ligne,

ani = FuncAnimation(fig, update, frames=len(t_imp), interval=30, blit=True)
plt.show()



''' Simulation B : Neutron '''

d = 1
z0 = 10

V = lambda x: np.array(x)
psi0 = lambda z: (np.pi*d*d)**(-0.25)*np.exp(-(z-z0)**2/(2*d*d))

z, t, psi = schrodinger_semi_imp(0,40,600,600,30, V, psi0)
dz = z[1] - z[0]
rho = np.abs(psi)**2

# Valeur moyenne de z
# Ici,  rho a la forme (m+1, n+1), z a la forme (m+1,) donc on lui ajoute une dimension et on somme sur la bonne
moy_z = np.sum(rho*z[:, None], axis=0) * dz

# Animation
fig, ax = plt.subplots(figsize=(8, 5))
ligne, = ax.plot(z, rho[:, 0] , 'b-', lw=2)

# Limites et légendes du graphe
ax.set_xlim(0, 25)
ax.set_ylim(0, np.max(rho) * 1.1)
ax.set_xlabel("z")
ax.set_ylabel(r"$|\psi(z,t)|^2$")
ax.grid(True)

# Mise à jour de chaque frame
def update(frame):
    ligne.set_ydata(rho[:, frame])
    ax.set_title(f"Rebonds du Neutron : t = {t[frame]:.2f}")
    return ligne,

ani = FuncAnimation(fig, update, frames=len(t), interval=30, blit=True)
plt.show()



""" Comparaison avec le rebond classique """

def rebond_classique(t, z0, g=1.0):
    tau = np.sqrt(2 * z0 / g)
    s = np.mod(t + tau, 2 * tau) - tau
    return z0 - 0.5 * g * s**2

plt.figure(figsize=(10, 5))
plt.plot(t, moy_z, 'b-', lw=2, label=r"Quantique $\langle z \rangle(t)$")
plt.plot(t, rebond_classique(t, z0), 'r--', lw=1.5, label="Classique (bille)")
plt.xlabel("t")
plt.ylabel("z")
plt.title(r"Hauteur moyenne du neutron $\langle z \rangle(t)$ vs rebond classique")
plt.legend()
plt.grid(True)
plt.show()
