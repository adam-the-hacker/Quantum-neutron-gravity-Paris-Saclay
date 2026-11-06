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
plt.plot(x_imp, rho[:, 0], label="t = 0 (départ)")
plt.plot(x_imp, rho[:, len(t_imp)//4], label="t = T/4")
plt.plot(x_imp, rho[:, len(t_imp)//2], label="t = T/2")
plt.plot(x_imp, rho[:, -1], label="t = T (fin)")
plt.xlabel("x")
plt.ylabel(r"$|\psi(x,t)|^2$")
plt.title("Profils de densité dans le puits infini")
plt.legend()
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

ani = FuncAnimation(fig, update, frames=len(t_imp), interval=30, blit=False)
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
# Ici, rho a la forme (m+1, n+1), z a la forme (m+1,) donc on lui ajoute une dimension et on somme sur la bonne
moy_z = np.sum(rho*z[:, None], axis=0) * dz


# Vérification du paquet initial
plt.figure(figsize=(4, 6))
plt.plot(psi0(z), z, 'b-', lw=2)
plt.axhline(z0, color='r', ls='--', label=f'z0 = {z0}')
plt.xlabel(r'$\psi_0(z)$')
plt.ylabel('z (hauteur)')
plt.title('Paquet initial vertical')
plt.grid(True)
plt.legend()
plt.show()


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

ani = FuncAnimation(fig, update, frames=len(t), interval=30, blit=False)
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



""" Calcul de Delta(p), v0 = <p>(0), delta(z) et inégalité de Heisenberg """

# <z^2> et delta(z)
moy_z2 = np.sum(rho*z[:, None]*z[:, None], axis=0) * dz
delta_z = (moy_z2 - moy_z**2)**0.5

# <p>0, <p>, <p^2> et delta(p)
# Opérateur p = -i d/dz
# On utilise la dérivée dpsi/dz = psi(j+1) - psi(j-1) / 2dz
p_psi = np.zeros_like(psi)
p_psi[1:-1] = -1j*(psi[2:] - psi[:-2])/(2*dz)


def moy(psi):
    # On somme sur l'axe spatial axis=0
    return np.real(np.sum(np.conj(psi)*p_psi, axis=0)*dz)

v0 = moy(psi)[0]
print(v0)
# <p^2> = <psi|p^2|psi> = <p* psi | p psi> = |p psi|^2 = |dpsi/dz|^2
delta_p = (np.sum(np.abs(p_psi)**2, axis=0)*dz - moy(psi)**2)**0.5

# Heisenberg adimensionnée avec marge d'erreur (dû aux différences finies discrètes)
# np.all(array) renvoie true si tout le tableau de booléens est vrai
print(np.all(delta_p*delta_z > 0.5 - 1e-3))
assert (np.all(delta_p*delta_z > 0.5 - 1e-3)), "Heisenberg non respecté"

plt.figure(figsize=(10, 4))
plt.plot(t, delta_z * delta_p, 'purple', lw=2, label=r'$\Delta z(t) \cdot \Delta p(t)$')
plt.axhline(0.5, color='black', ls='--', lw=1.5, label=r'Limite théorique $\hbar/2 = 0.5$')

plt.xlabel('t')
plt.ylabel(r'$\Delta z \cdot \Delta p$')
plt.title(r"Vérification de l'inégalité de Heisenberg")
plt.legend()
plt.grid(True)
plt.show()
