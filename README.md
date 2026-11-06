# Quantum Rebounds of Neutrons in a Gravitational Field
## Unsteady Simulations of the Schrödinger Equation

Numerical solver for the 1D time-dependent Schrödinger equation (TDSE) modeling ultracold neutrons in a gravitational potential above an ideal mirror.

## 1. Model & Adimensionalization

Dimensionless TDSE ($\hbar = m = g = 1$):

$$
i \partial_t \psi = -\frac{1}{2} \partial_{zz} \psi + V(z)\psi
$$

Characteristic physical scales:
* Length scale: $L_* = (\hbar^2 / m^2 g)^{1/3} \approx 5.87 \ \mu\text{m}$
* Time scale: $T_* = (m\hbar / g^2)^{1/3} \approx 1.09 \text{ ms}$
* Energy scale: $E_* = (m g^2 \hbar^2)^{1/3} \approx 0.60 \text{ peV}$

Boundary conditions: homogeneous Dirichlet $\psi(a, t) = \psi(b, t) = 0$.

## 2. Numerical Schemes

### Discretization
* Space: uniform grid $z_j = a + j \Delta z$ with $j \in [0, M]$ and $\Delta z = (b-a)/M$.
* Time: uniform steps $t_n = n \Delta t$ with $n \in [0, N]$ and $\Delta t = T/N$.
* Spatial operator (2nd-order centered differences):

$$
\partial_{zz} \psi_j^n \approx \frac{\psi_{j+1}^n - 2\psi_j^n + \psi_{j-1}^n}{\Delta z^2}
$$

### Time Stepping
**1. Explicit Euler (`schrodinger_exp`):**

$$
\psi_j^{n+1} = \psi_j^n + i \frac{\Delta t}{2\Delta z^2} (\psi_{j+1}^n - 2\psi_j^n + \psi_{j-1}^n) - i \Delta t V_j \psi_j^n
$$

Unconditionally unstable for Schrödinger equations; requires $\Delta t \ll \Delta z^2 / 2$ ($N \ge 20000$) to delay numerical blowup.

**2. Crank-Nicolson (`schrodinger_semi_imp`):**

$$
A \cdot \psi^{n+1} = B \cdot \psi^n
$$

with $A$ and $B$ tridiagonal complex matrices satisfying $B = -A^*$:

$$
A_{j,j} = \frac{i}{\Delta t} - \frac{1}{2\Delta z^2} - \frac{V_j}{2}, \qquad A_{j, j\pm 1} = \frac{1}{4\Delta z^2}
$$

Unconditionally stable and strictly unitary ($\int |\psi|^2 dz = 1$ conserved to machine precision).

## 3. Implemented Simulations

### Case A: Infinite Potential Well (Benchmark)
* Domain: $x \in [0, 1]$ with $V(x) = 0$.
* Initial condition: $\psi_0(x) = \sqrt{30}x(1-x)$.
* Stability comparison: explicit ($N \ge 20000$) vs. implicit ($N = 200$).

### Case B: Neutron in Gravity
* Domain: $z \in [0, 40]$ with potential $V(z) = z$ ($M = 600, N = 600, T = 30$).
* Initial Gaussian wave packet ($z_0 = 10, d = 1$):

$$
\psi_0(z) = (\pi d^2)^{-1/4} \exp\left(-\frac{(z - z_0)^2}{2d^2}\right)
$$

### Computed Observables

**Expectation values:**

$$
\langle z \rangle(t_n) = \sum_{j} |\psi_j^n|^2 z_j \Delta z
$$

$$
\langle p \rangle(t_n) = \text{Re} \left[ \sum_{j} (\psi_j^n)^* \left(-i \frac{\psi_{j+1}^n - \psi_{j-1}^n}{2\Delta z}\right) \Delta z \right]
$$

**Standard deviations and Heisenberg uncertainty:**

$$
\Delta z(t) = \sqrt{\langle z^2 \rangle - \langle z \rangle^2}, \qquad \Delta p(t) = \sqrt{\langle p^2 \rangle - \langle p \rangle^2}
$$

Verification of $\Delta z(t) \cdot \Delta p(t) \ge 0.5$ over all time steps.

**Classical bouncing trajectory:**

$$
z_{\text{class}}(t) = z_0 - \frac{1}{2}g s(t)^2 \quad \text{with} \quad s(t) = (t + \tau) \bmod (2\tau) - \tau, \quad \tau = \sqrt{\frac{2z_0}{g}}
$$

## 4. Usage

Dependencies: `numpy`, `matplotlib`.

```bash
python simulation.py
```
