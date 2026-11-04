# Quantum Rebounds of Neutrons in a Gravitational Field
## Unsteady Simulations of the Schrödinger Equation
#%% 1. ENTREES ET LIBRAIRIES
############################
# Equation de Schrödinger adimensionnée (hbar = m = 1) par différences finies :
#       E phi = -1/2 phi'' + V phi            (stationnaire)
#       i dpsi/dt = -1/2 psi'' + V psi        (instationnaire, Crank-Nicolson)
# Conditions de Dirichlet : psi = 0 aux deux bords du domaine [a,b].
#
#   PARTIE I   : états propres (9 potentiels dont puits fini, marche, ...)
#   PARTIE II  : effet tunnel stationnaire (double puits)
#   PARTIE III : évolution temporelle SEMI-IMPLICITE (comme DF3) :
#                barrière, marche, puits fini, neutron

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh, solve_banded
from scipy.optimize import brentq

##############################################################################
#                     PARTIE I : PROBLEME STATIONNAIRE
##############################################################################
#%% 2. Solveur stationnaire
##########################
def etats_propres(a, b, M, Vdef, nb):
    """x (M+1 points), E (nb premières énergies), phi (M+1, nb) normalisées."""
    dx = (b - a) / M
    c = 1 / dx**2
    x = np.linspace(a, b, M + 1)
    V = Vdef(x)

    n = M - 1                                 # points intérieurs
    H = np.zeros([n, n])
    for k in range(n):
        H[k, k] = c + V[k + 1]
        if k > 0:
            H[k, k - 1] = -c / 2
        if k < n - 1:
            H[k, k + 1] = -c / 2

    E, phi_int = eigh(H)                      # H symétrique : valeurs déjà triées
    E = E[:nb]
    phi = np.zeros([M + 1, nb])
    phi[1:M, :] = phi_int[:, :nb]
    for k in range(nb):
        phi[:, k] /= np.sqrt(np.sum(phi[:, k]**2) * dx)
        i0 = np.argmax(np.abs(phi[:, k]) > 1e-3 * np.abs(phi[:, k]).max())
        if phi[i0, k] < 0:
            phi[:, k] *= -1
    return x, E, phi

#%% 3. Les potentiels stationnaires
##################################
def marche_sur_noeuds(x, V, xm, Vgauche, Vdroite):
    """Pour un potentiel discontinu en xm : valeur moyenne sur le noeud xm."""
    V[np.isclose(x, xm)] = (Vgauche + Vdroite) / 2
    return V

# 1) puits infini (domaine [0,1])
def V_puits_infini(x):
    return np.zeros_like(x)

# 2) oscillateur harmonique   V = x²/2
def V_oscillateur(x):
    return 0.5 * x**2

# 3) puits FINI symétrique : V=0 pour |x|<l ,  V=V0 ailleurs
l_puits, V0_puits = 1.0, 10.0
def V_puits_fini(x):
    V = np.where(np.abs(x) < l_puits, 0.0, V0_puits)
    for xm in (-l_puits, l_puits):
        marche_sur_noeuds(x, V, xm, 0.0, V0_puits)
    return V

# 4) puits FINI asymétrique : V=V_G (x<-l) , 0 (|x|<l) , V_D (x>l)
V_G, V_D = 10.0, 5.0
def V_puits_asym(x):
    V = np.where(x < -l_puits, V_G, np.where(x > l_puits, V_D, 0.0))
    marche_sur_noeuds(x, V, -l_puits, V_G, 0.0)
    marche_sur_noeuds(x, V, +l_puits, 0.0, V_D)
    return V

# 5) double puits   V = (x²-4)²/8
def V_double_puits(x):
    return (x**2 - 4)**2 / 8

# 6) Pöschl-Teller   V = -l(l+1)/(2 cosh²x) ,  E_n = -(l-n)²/2
lam = 3
def V_poschl_teller(x):
    return -lam * (lam + 1) / (2 * np.cosh(x)**2)

# 7) Morse   V = D (1-exp(-alpha x))²
D_morse, alpha = 8.0, 1.0
def V_morse(x):
    return D_morse * (1 - np.exp(-alpha * x))**2

# 8) neutron dans le champ de gravité   V = z  (mur en z = 0)
def V_gravite(x):
    return x.copy()

# 9) marche de potentiel : V=0 pour x<0 , V=V0 pour x>0  (dans une boîte)
V0_marche = 4.0
def V_marche(x):
    V = np.where(x > 0, V0_marche, 0.0)
    return marche_sur_noeuds(x, V, 0.0, 0.0, V0_marche)

#%% 4. Solutions exactes (quand on les connaît)
##############################################
# Puits fini (symétrique ou non), de largeur L, murs V1 (gauche) et V2 (droite) :
# condition de raccordement  k L + arctan(k/kappa1) + arctan(k/kappa2) = n pi
# avec k = sqrt(2E),  kappa_i = sqrt(2(V_i - E)),  n = 1, 2, ...
def E_exact_puits(V1, V2, L, nb):
    Emax = min(V1, V2)
    def g(E, n):
        k = np.sqrt(2 * E)
        return (k * L + np.arctan(k / np.sqrt(2 * (V1 - E)))
                + np.arctan(k / np.sqrt(2 * (V2 - E))) - n * np.pi)
    racines = []
    for n in range(1, nb + 1):
        lo, hi = 1e-9, Emax - 1e-9
        if g(hi, n) <= 0:          # plus d'état lié
            break
        racines.append(brentq(g, lo, hi, args=(n,)))
    return np.array(racines)

airy = np.array([2.33811, 4.08795, 5.52056, 6.78671, 7.94413, 9.02265])  # |a_n|

def E_morse(nb):
    w, n = alpha * np.sqrt(2 * D_morse), np.arange(nb)
    return w * (n + .5) - (w * (n + .5))**2 / (4 * D_morse)

#%% 5. Paramètres des cas stationnaires
######################################
cas = {
 'puits_infini':  dict(a=0,   b=1,  M=300,  nb=4, V=V_puits_infini,
                       E_ex=lambda nb: (np.arange(1, nb + 1) * np.pi)**2 / 2),
 'oscillateur':   dict(a=-8,  b=8,  M=800,  nb=5, V=V_oscillateur,
                       E_ex=lambda nb: np.arange(nb) + 0.5),
 'puits_fini':    dict(a=-6,  b=6,  M=1200, nb=4, V=V_puits_fini,
                       E_ex=lambda nb: E_exact_puits(V0_puits, V0_puits, 2 * l_puits, nb)),
 'puits_asym':    dict(a=-6,  b=6,  M=1200, nb=4, V=V_puits_asym,
                       E_ex=lambda nb: E_exact_puits(V_G, V_D, 2 * l_puits, nb)),
 'double_puits':  dict(a=-6,  b=6,  M=800,  nb=6, V=V_double_puits, E_ex=None),
 'poschl_teller': dict(a=-15, b=15, M=1500, nb=3, V=V_poschl_teller,
                       E_ex=lambda nb: -(lam - np.arange(nb))**2 / 2),
 'morse':         dict(a=-2,  b=10, M=1200, nb=4, V=V_morse, E_ex=E_morse),
 'gravite':       dict(a=0,   b=30, M=600,  nb=5, V=V_gravite,
                       E_ex=lambda nb: airy[:nb] / 2**(1 / 3)),
 'marche':        dict(a=-8,  b=8,  M=800,  nb=6, V=V_marche, E_ex=None),
}

#%% 6. Choisir un cas stationnaire, résoudre, comparer, tracer
#############################################################
choix = 'puits_asym'          # <-- clés possibles : voir le dictionnaire `cas`

p = cas[choix]
x, E, phi = etats_propres(p['a'], p['b'], p['M'], p['V'], p['nb'])
V = p['V'](x)
nb = p['nb']

print(f"Cas stationnaire : {choix}")
if p['E_ex'] is not None:
    E_ex = p['E_ex'](nb)
    print(" n   E numérique   E exacte    erreur relative")
    for k in range(len(E_ex)):
        print(f" {k}  {E[k]:11.5f}  {E_ex[k]:11.5f}   {abs(E[k]-E_ex[k])/abs(E_ex[k]):.2e}")
    if len(E_ex) < nb:
        print(f" (seuls {len(E_ex)} états liés existent ; les suivants sont des "
              f"états de la boîte [a,b])")
else:
    print(" n   E numérique")
    for k in range(nb):
        print(f" {k}  {E[k]:11.5f}")

amp = 0.8 * np.min(np.diff(E)) / np.abs(phi).max()
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(x, V, 'k', lw=2, label='V(x)')
for k in range(nb):
    ax[0].axhline(E[k], color='gray', ls=':', lw=.8)
    ax[0].plot(x, E[k] + amp * phi[:, k], label=f'n={k}, E={E[k]:.3f}')
ax[0].set_ylim(V.min() - 0.5, E[-1] + 1.5 * np.min(np.diff(E)))
ax[0].set_xlabel('x'); ax[0].set_ylabel('E'); ax[0].grid(); ax[0].legend(fontsize=8)
ax[0].set_title(f'{choix} : niveaux et fonctions propres')
for k in range(nb):
    ax[1].plot(x, phi[:, k]**2, label=f'n={k}')
ax[1].set_xlabel('x'); ax[1].set_ylabel(r'$|\phi_n|^2$'); ax[1].grid(); ax[1].legend()
ax[1].set_title('Densités de probabilité')
plt.show()

##############################################################################
#                PARTIE II : EFFET TUNNEL STATIONNAIRE (DOUBLE PUITS)
##############################################################################
#%% 7. Doublet E0/E1 et oscillation tunnel
#########################################
# V = h (x²-a²)²/a⁴ : minima en ±a, barrière centrale de hauteur h
a_dp = 2.0
def double_puits_h(h):
    return lambda x: h * (x**2 - a_dp**2)**2 / a_dp**4

hs = np.array([1, 2, 3, 4, 6, 8, 10, 12])
dE = np.array([np.diff(etats_propres(-6, 6, 600, double_puits_h(h), 2)[1])[0] for h in hs])
print("\nDoublet : h    E1-E0       demi-période pi/(E1-E0)")
for h, d_ in zip(hs, dE):
    print(f"          {h:3d}  {d_:10.3e}   {np.pi/d_:10.1f}")

h = 4.0
x, E, phi = etats_propres(-6, 6, 600, double_puits_h(h), 2)
dx = x[1] - x[0]
i_d = np.argmin(np.abs(x - a_dp))
phi[:, 1] *= np.sign(phi[i_d, 1] * phi[i_d, 0])      # phi0, phi1 de même signe à droite
Tt = np.pi / (E[1] - E[0])
t = np.linspace(0, 2 * Tt, 400)
# psi(0) = (phi0 + phi1)/sqrt(2) : localisé dans le puits de droite
psi = (np.outer(phi[:, 0], np.exp(-1j * E[0] * t)) +
       np.outer(phi[:, 1], np.exp(-1j * E[1] * t))) / np.sqrt(2)
PD = np.sum(np.abs(psi[x > 0, :])**2, axis=0) * dx
print(f"h={h} : P_droite(0) = {PD[0]:.4f} ; P_droite(pi/(E1-E0)) = "
      f"{PD[np.argmin(np.abs(t - Tt))]:.4f}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].semilogy(hs, dE, 'o-'); ax[0].grid()
ax[0].set_xlabel('hauteur de barrière h'); ax[0].set_ylabel('E1 - E0')
ax[0].set_title("Doublet : écart exponentiellement petit")
ax[1].plot(t, PD, label='puits de droite'); ax[1].plot(t, 1 - PD, label='puits de gauche')
ax[1].set_xlabel('t'); ax[1].set_ylabel('probabilité'); ax[1].legend(); ax[1].grid()
ax[1].set_title('Oscillation tunnel')
for tt in [0, Tt / 2, Tt]:
    n = np.argmin(np.abs(t - tt))
    ax[2].plot(x, np.abs(psi[:, n])**2, label=f't={t[n]:.0f}')
ax[2].plot(x, double_puits_h(h)(x) / 40, 'k--', lw=.8, label='V (réduit)')
ax[2].set_xlabel('x'); ax[2].set_ylabel(r'$|\psi|^2$'); ax[2].legend(); ax[2].grid()
plt.show()

##############################################################################
#         PARTIE III : EVOLUTION TEMPORELLE (SCHEMA SEMI-IMPLICITE)
##############################################################################
#%% 8. Solveur instationnaire SEMI-IMPLICITE (comme Lorenz / Burgers dans DF3)
###############################################################################
# On écrit  dpsi/dt = L psi + NL(t,psi)  avec
#       L psi   = -i K psi = (i/2) d²psi/dx²     terme LINEAIRE  -> Crank-Nicolson (implicite)
#       NL      = -i V(x,t) psi                  terme potentiel -> explicite
#
# Comme pour Burgers (CN pour la diffusion + Euler explicite pour NL), on obtient
#       Og psi^{n+1} = Od psi^n + dt NL^n
# avec r = i dt / (4 dx²) et, sur les points intérieurs,
#       Og = tridiag[-r, 1+2r, -r]        Od = tridiag[+r, 1-2r, +r]
# Og ne change jamais : même si V dépend du temps, il suffit de ré-évaluer NL^n.
#
# Deux façons de traiter le terme explicite (argument `pot`) :
#   pot='euler' : NL^n = -i V^n psi^n   (exactement comme DF3). |1 - i V dt| > 1 :
#                 la norme croît à chaque pas, ~exp(V_max² dt T / 2) -> il faut
#                 V_max² dt T << 1 sur TOUT le domaine (dt très petit).
#   pot='phase' : psi~ = exp(-i V dt) psi^n  puis  Og psi^{n+1} = Od psi~.
#                 Même principe (potentiel explicite, cinétique implicite) mais le
#                 facteur de phase est de module 1 : norme conservée, stable.
# schema='CN' : Crank-Nicolson complet (V stationnaire implicite), pour comparer.
#
# On ne stocke que N_snap+1 états (cf. exercice 3.6.2 de DF3).
# Og est tridiagonale : on la résout avec solve_banded au lieu de np.linalg.solve
# (même résultat, beaucoup plus rapide : O(M) au lieu de O(M³) par pas).
def evolution(a, b, M, N, T, Vdef, psi0def, N_snap=300,
              schema='semi', pot='phase', V_instationnaire=False):
    """Renvoie x (M+1), t_snap (N_snap+1), psi (M+1, N_snap+1) complexe.
    Vdef(x) si stationnaire, Vdef(x,t) si V_instationnaire=True (schema 'semi')."""
    if N % N_snap != 0:
        raise ValueError("N doit être un multiple de N_snap")
    dx = (b - a) / M
    dt = T / N
    r = 1j * dt / (4 * dx**2)
    x = np.linspace(a, b, M + 1)
    n_save = N // N_snap

    # --- matrice de gauche Og en format "bande" (3 diagonales), bords : 1
    ab = np.zeros((3, M + 1), dtype=complex)
    ab[1, :] = 1 + 2 * r              # diagonale principale
    ab[0, 2:] = -r                    # sur-diagonale
    ab[2, :M - 1] = -r                # sous-diagonale
    ab[1, 0] = 1; ab[1, M] = 1        # points de bord : psi = 0
    if schema == 'CN':                # CN complet : le potentiel passe à gauche
        if V_instationnaire:
            raise ValueError("schema='CN' : V doit être stationnaire")
        ab[1, 1:M] += 1j * dt / 2 * Vdef(x)[1:M]
    elif schema != 'semi':
        raise ValueError("schema doit être 'semi' ou 'CN'")

    psi = np.asarray(psi0def(x), dtype=complex).copy()   # COMPLEXE !
    psi[0] = psi[M] = 0
    psi_snap = np.zeros([M + 1, N_snap + 1], dtype=complex)
    psi_snap[:, 0] = psi
    t_snap = np.linspace(0, T, N_snap + 1)
    V = None if V_instationnaire else Vdef(x)

    for n in range(N):
        if schema == 'semi':
            Vn = Vdef(x, n * dt) if V_instationnaire else V      # V(x,t_n)
            u = np.exp(-1j * Vn * dt) * psi if pot == 'phase' else psi
        else:
            u = psi

        # second membre  Od u  (+ terme NL pour pot='euler')
        rhs = np.zeros(M + 1, dtype=complex)
        rhs[1:M] = u[1:M] + r * (u[0:M - 1] - 2 * u[1:M] + u[2:M + 1])
        if schema == 'semi' and pot == 'euler':
            rhs[1:M] += -1j * dt * Vn[1:M] * psi[1:M]
        elif schema == 'CN':
            rhs[1:M] -= 1j * dt / 2 * V[1:M] * psi[1:M]

        # système linéaire  Og psi^{n+1} = rhs
        psi = solve_banded((1, 1), ab, rhs)

        if (n + 1) % n_save == 0:
            psi_snap[:, (n + 1) // n_save] = psi
        if not np.isfinite(psi).all() or np.max(np.abs(psi)) > 1e6:
            print(f"Instabilité numérique : boum au pas {n} (essayer pot='phase' "
                  f"ou diminuer dt)")
            break
    return x, t_snap, psi_snap

#%% 9. Potentiels et paquets d'onde instationnaires
##################################################
def paquet(x0, d, k0):
    """Gaussienne normalisée centrée en x0, largeur d, impulsion k0."""
    return lambda x: (np.pi * d**2)**(-0.25) * np.exp(-(x - x0)**2 / (2 * d**2)) \
                     * np.exp(1j * k0 * x)

# --- barrière rectangulaire (effet tunnel)
V0_bar, w_bar = 6.0, 0.5
def V_barriere(x):
    V = np.where(np.abs(x) < w_bar / 2, V0_bar, 0.0)
    for xm in (-w_bar / 2, w_bar / 2):
        marche_sur_noeuds(x, V, xm, 0.0, V0_bar)
    return V

# --- marches de potentiel : 0 pour x<0, V0 pour x>0
def marche_V0(V0):
    def V(x):
        Vx = np.where(x > 0, V0, 0.0)
        return marche_sur_noeuds(x, Vx, 0.0, 0.0, V0)
    return V

# --- puits fini (V=0 si |x|<l, V0 ailleurs) : réutilise V_puits_fini
# --- neutron : V = z  (réutilise V_gravite)

def T_barriere(E):
    """Transmission exacte d'une barrière rectangulaire V0_bar, largeur w_bar."""
    E = np.atleast_1d(np.asarray(E, float)); out = np.empty_like(E)
    bas = E < V0_bar
    ka = np.sqrt(2 * (V0_bar - E[bas]))
    out[bas] = 1 / (1 + V0_bar**2 * np.sinh(ka * w_bar)**2 / (4 * E[bas] * (V0_bar - E[bas])))
    q = np.sqrt(2 * (E[~bas] - V0_bar))
    out[~bas] = 1 / (1 + V0_bar**2 * np.sin(q * w_bar)**2 / (4 * E[~bas] * (E[~bas] - V0_bar)))
    return out

def T_marche(E, V0):
    """Transmission exacte d'une marche : 0 si E<V0, 4 k1 k2/(k1+k2)² sinon."""
    E = np.atleast_1d(np.asarray(E, float)); out = np.zeros_like(E)
    h = E > V0
    k1, k2 = np.sqrt(2 * E[h]), np.sqrt(2 * (E[h] - V0))
    out[h] = 4 * k1 * k2 / (k1 + k2)**2
    return out

def moyenne_sur_paquet(Tfun, d, k0):
    """Moyenne de T(E=k²/2) pondérée par |psi~(k)|² ~ exp(-d²(k-k0)²)."""
    k = np.linspace(0.05, k0 + 6 / d + 3, 6000)
    w = np.exp(-d**2 * (k - k0)**2)
    return np.sum(w * Tfun(k**2 / 2)) / np.sum(w)

#%% 10. Paramètres des cas instationnaires
#########################################
cas_t = {
 'barriere': dict(type='barriere', a=-40, b=40, M=800, N=3000, N_snap=300, T=12.0,
                  V=V_barriere, psi0=paquet(-15, 2.0, 3.0), d=2.0, k0=3.0),
 'marche_E>V0': dict(type='marche', a=-40, b=40, M=800, N=3000, N_snap=300, T=12.0,
                  V0=2.0, V=marche_V0(2.0), psi0=paquet(-15, 2.0, 3.0), d=2.0, k0=3.0),
 'marche_E<V0': dict(type='marche', a=-40, b=40, M=800, N=3000, N_snap=300, T=12.0,
                  V0=7.0, V=marche_V0(7.0), psi0=paquet(-15, 2.0, 3.0), d=2.0, k0=3.0),
 # paquet très étroit au centre d'un puits fini : une partie s'échappe
 'puits_fini': dict(type='puits', a=-30, b=30, M=1200, N=3000, N_snap=300, T=6.0,
                  V=V_puits_fini, psi0=paquet(0.0, 0.2, 0.0)),
 # neutron qui tombe et rebondit (unités naturelles, V = z)
 'neutron': dict(type='neutron', a=0, b=40, M=800, N=6000, N_snap=300, T=27.0, z0=10.0,
                  V=V_gravite, psi0=lambda z: (np.pi**-0.25) * np.exp(-(z - 10.0)**2 / 2)),
}

def rebond_classique(t, z0, g=1.0):
    tau = np.sqrt(2 * z0 / g)
    s = np.mod(t + tau, 2 * tau) - tau
    return z0 - 0.5 * g * s**2

#%% 11. Choisir un cas instationnaire, simuler, analyser
#######################################################
choix_t = 'marche_E>V0'     # <-- 'barriere','marche_E>V0','marche_E<V0','puits_fini','neutron'
schema = 'semi'             # 'semi' (semi-implicite) ou 'CN' (Crank-Nicolson complet)
pot = 'phase'               # traitement explicite du potentiel : 'phase' (stable) ou 'euler' (DF3)

q = cas_t[choix_t]
x, t, psi = evolution(q['a'], q['b'], q['M'], q['N'], q['T'], q['V'], q['psi0'],
                      N_snap=q['N_snap'], schema=schema, pot=pot)
dx = x[1] - x[0]
V = q['V'](x)
rho = np.abs(psi)**2
norme = np.sum(rho, axis=0) * dx
print(f"\nCas instationnaire : {choix_t}  schéma={schema}, pot={pot}, dt={q['T']/q['N']:.4f}"
      f"   (norme min/max = {norme.min():.6f} / {norme.max():.6f})")
# comparaison avec Crank-Nicolson complet (référence)
_, _, psi_ref = evolution(q['a'], q['b'], q['M'], q['N'], q['T'], q['V'], q['psi0'],
                          N_snap=q['N_snap'], schema='CN')
print(f"  écart max sur |psi|² par rapport à CN complet : "
      f"{np.max(np.abs(rho - np.abs(psi_ref)**2)):.2e}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
if q['type'] in ('barriere', 'marche'):
    if q['type'] == 'barriere':
        zone_g, zone_d = x < -w_bar / 2, x > w_bar / 2
        Tfun = T_barriere
    else:
        zone_g, zone_d = x < 0, x > 0
        Tfun = lambda E: T_marche(E, q['V0'])
    Pg = np.sum(rho[zone_g, :], axis=0) * dx
    Pd = np.sum(rho[zone_d, :], axis=0) * dx
    T_ex = moyenne_sur_paquet(Tfun, q['d'], q['k0'])
    print(f"  réfléchi = {Pg[-1]:.4f}  transmis = {Pd[-1]:.4f}")
    print(f"  T exact moyenné sur le paquet = {T_ex:.4f}   "
          f"(E moyenne = {q['k0']**2/2}, V0 = {q.get('V0', V0_bar)})")
    for tt in [0, 4, 6, 8, q['T']]:
        n = np.argmin(np.abs(t - tt)); ax[0].plot(x, rho[:, n], label=f't={t[n]:.0f}')
    ax[0].set_xlim(-30, 30)
    ax[1].plot(t, Pg, label='gauche (réfléchi)'); ax[1].plot(t, Pd, label='droite (transmis)')
    ax[1].axhline(T_ex, color='gray', ls='--', label='T exact (moyenné)')
elif q['type'] == 'puits':
    Pin = np.sum(rho[np.abs(x) < l_puits, :], axis=0) * dx
    print(f"  probabilité dans le puits : t=0 -> {Pin[0]:.4f} ; t=T -> {Pin[-1]:.4f} "
          f"(le reste s'est échappé : impossible dans un puits infini)")
    for tt in [0, 1, 2, 4, q['T']]:
        n = np.argmin(np.abs(t - tt)); ax[0].plot(x, rho[:, n], label=f't={t[n]:.1f}')
    ax[0].set_xlim(-12, 12)
    ax[1].plot(t, Pin, label='dans le puits'); ax[1].plot(t, 1 - Pin, label='hors du puits')
elif q['type'] == 'neutron':
    zmoy = (x[:, None] * rho).sum(axis=0) * dx
    ax[0].plot(t, zmoy, label=r'quantique $\langle z\rangle$')
    ax[0].plot(t, rebond_classique(t, q['z0']), '--', label='classique')
    ax[0].set_xlabel('t'); ax[0].set_ylabel('z')
    for tt in [0, 5, 10, 20, q['T']]:
        n = np.argmin(np.abs(t - tt)); ax[1].plot(x, rho[:, n], label=f't={t[n]:.0f}')
    print(f"  <z>(0) = {zmoy[0]:.3f}, <z> max/min = {zmoy.max():.3f} / {zmoy.min():.3f}")
if q['type'] != 'neutron':
    ax[0].set_xlabel('x'); ax[0].set_ylabel(r'$|\psi|^2$')
    ax[1].set_xlabel('t'); ax[1].set_ylabel('probabilité')
else:
    ax[1].set_xlabel('z'); ax[1].set_ylabel(r'$|\psi|^2$')
for a_ in ax:
    a_.legend(); a_.grid()
fig.suptitle(choix_t)
plt.show()

#%% 12. Animation de |psi|² (le potentiel est dessiné à échelle réduite)
#######################################################################
fig, axa = plt.subplots(figsize=(9, 4))
ymax = 1.1 * rho.max()
V_aff = 0.4 * ymax * (V - V.min()) / max(np.ptp(V), 1e-12)
ligne, = axa.plot(x, rho[:, 0], 'b')
axa.fill_between(x, 0, V_aff, color='gray', alpha=.4, label='V (échelle réduite)')
axa.set_xlim(x[0], x[-1])
if q['type'] in ('barriere', 'marche'):
    axa.set_xlim(-30, 30)
if q['type'] == 'puits':
    axa.set_xlim(-12, 12)
axa.set_ylim(0, ymax); axa.set_xlabel('x'); axa.grid(); axa.legend(loc='upper right')
def maj(n):
    ligne.set_ydata(rho[:, n]); axa.set_title(f'{choix_t} : t = {t[n]:.2f}')
    return ligne,
ani = FuncAnimation(fig, maj, frames=len(t), interval=40)
plt.show()

#%% 13. (Optionnel) Tout tester d'un coup
########################################
print("\n=== Cas stationnaires ===")
for nom, c_ in cas.items():
    xx, EE, _ = etats_propres(c_['a'], c_['b'], c_['M'], c_['V'], c_['nb'])
    if c_['E_ex'] is not None:
        Eex = c_['E_ex'](c_['nb'])
        err = np.max(np.abs(EE[:len(Eex)] - Eex) / np.abs(Eex))
        print(f"{nom:14s} E0 = {EE[0]:9.5f}   erreur relative max = {err:.1e}")
    else:
        print(f"{nom:14s} E0 = {EE[0]:9.5f}   E1 = {EE[1]:9.5f}   (pas de solution exacte)")

print("\n=== Cas instationnaires ===")
for nom, c_ in cas_t.items():
    xx, tt_, pp = evolution(c_['a'], c_['b'], c_['M'], c_['N'], c_['T'], c_['V'], c_['psi0'],
                            N_snap=c_['N_snap'], schema='semi', pot='phase')
    r = np.abs(pp)**2; d_x = xx[1] - xx[0]
    if c_['type'] == 'marche':
        Tex = moyenne_sur_paquet(lambda E: T_marche(E, c_['V0']), c_['d'], c_['k0'])
        print(f"{nom:14s} transmis = {np.sum(r[xx>0,-1])*d_x:.4f}   T exact moyenné = {Tex:.4f}")
    elif c_['type'] == 'barriere':
        Tex = moyenne_sur_paquet(T_barriere, c_['d'], c_['k0'])
        print(f"{nom:14s} transmis = {np.sum(r[xx>w_bar/2,-1])*d_x:.4f}   T exact moyenné = {Tex:.4f}")
    elif c_['type'] == 'puits':
        print(f"{nom:14s} proba dans le puits à t=T : {np.sum(r[np.abs(xx)<l_puits,-1])*d_x:.4f}")
    else:
        zm = (xx[:, None] * r).sum(axis=0) * d_x
        print(f"{nom:14s} <z> varie entre {zm.min():.2f} et {zm.max():.2f}")
[In progress, will be published when the project is complete]
