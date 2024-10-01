
# Beam Simulation

The flexure-based design of MAGPIE's sensing mechanism is simulated using Von Kármán beam theory to account for large deflections. This model is vital for predicting how the flexure will deform under applied forces.

**Mathematical Model:**

The vertical force balance (bending equation):

$$
EI \frac{d^4 w(x)}{dx^4} = q(x) - \frac{d}{dx} \left( N(x) \frac{dw(x)}{dx} \right)
$$

Where
- $E I$ is the flexural rigidity
- $w(x)$ is the vertical deflection
- $N(x)$ is the axial force
- $q(x)$ is the distributed transverse load

For horizontal force balance (stretching equation):

$$
\frac{d}{d x}\left(E A \frac{d u(x)}{d x}\right)=\frac{1}{2} E I\left(\frac{d w(x)}{d x}\right)^2
$$

This model helps ensure that the flexure design can withstand large deformations without failure while maintaining accurate force sensing.
