"""
Reproducing

"Enhancement and suppression of tunneling by controlling symmetries of a potential
barrie"
https://arxiv.org/abs/1006.0905
"""
from QuantumClassicalDynamics.split_op_schrodinger2D import SplitOpSchrodinger2D, np, ne
from QuantumClassicalDynamics.mub_qhamiltonian import MUBQHamiltonian

# load tools for creating animation
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation, writers


class VisualizeDynamics2D:
    """
    Visualize the tunnelling dynamics in 2D described in
        https://arxiv.org/abs/1006.0905

    Convention:
       X1 -- the interparticle degree of freedom (rho)
       X2 -- the center of mass degree of freedom (R)
    """
    def __init__(self, fig):
        """
        Initialize all propagators and frame
        :param fig: matplotlib figure object
        """
        #  Initialize systems
        self.set_quantum_sys()

        #################################################################
        #
        # Initialize plotting facility
        #
        #################################################################

        self.fig = fig

        ax = fig.add_subplot(111)

        ax.set_title('Wavefunction density, $| \\Psi(\\rho, R, t) |^2$')
        self._extent = [
            self.quant_sys.X2.min(),
            self.quant_sys.X2.max(),
            self.quant_sys.X1.min(),
            self.quant_sys.X1.max()
        ]
        self._aspect = (self._extent[1] - self._extent[0]) / (self._extent[3] - self._extent[2])

        self.img = ax.imshow(
            [[]],
            extent=self._extent,
            aspect=self._aspect,
            origin='lower',
            norm=LogNorm(vmin=1e-12, vmax=0.1)
        )

        self.fig.colorbar(self.img)

        ax.set_xlabel('$R$ (a.u.)')
        ax.set_ylabel('$\\rho$ (a.u.)')

    def set_quantum_sys(self):
        """
        Initialize quantum propagator
        :param self:
        :return:
        """
        ###################################################################
        #
        #  Diagonalize the inter particle degree of freedom Hamiltonian
        #  given in Eqs. (5) and (6)
        #
        ###################################################################
        rN = 1.961 # Determines the number of bound states in the inter particle degree of freedom
        rho_gridDIM = 512
        rho_amplitude = 60.
        U = "-2. * exp(-{rho} ** 2 / rN ** 2)"
        diff_U = "4. * exp(-{rho} ** 2 / rN ** 2) * {rho} / rN ** 2"

        interparticle_sys = MUBQHamiltonian(
            X_gridDIM=rho_gridDIM,
            X_amplitude=rho_amplitude,
            rN=rN,
            V=U.format(rho="X"),
            K="P ** 2"
        )

        print("Energies of the interparticle degree of freedom: ", interparticle_sys.get_energy(slice(0, 4)))

        ###################################################################
        #
        #  The system Hamiltonian is given by Eq. (5)
        #
        ###################################################################
        V = "{X} * exp(-{X} ** 2)"
        diff_V = "exp(-{X} ** 2) * (1. -2. * {X} ** 2)"

        self.quant_sys = SplitOpSchrodinger2D(
            t=0.,
            dt=0.01,
            X1_gridDIM=rho_gridDIM,
            X1_amplitude=rho_amplitude,
            X2_gridDIM=1024,
            X2_amplitude=70.,

            # kinetic energy part of the hamiltonian
            K="P2 ** 2 / 4. + P1 ** 2",

            # these functions are used for evaluating the Ehrenfest theorems
            diff_K_P1="2. * P1",
            diff_K_P2="0.5 * P2",

            # potential energy part of the hamiltonian
            rN=rN,
            alpha=3,
            V="alpha * ({}) + 3 * ({}) + {}".format(
                V.format(X="(X2 - 0.5 * X1)"),
                V.format(X="(X2 + 0.5 * X1)"),
                U.format(rho="X1"),
            ),

            # these functions are used for evaluating the Ehrenfest theorems
            diff_V_X1="-0.5 * alpha * ({}) + 1.5 * ({}) + {}".format(
                diff_V.format(X="(X2 - 0.5 * X1)"),
                diff_V.format(X="(X2 + 0.5 * X1)"),
                diff_U.format(rho="X1"),
            ),
            diff_V_X2="alpha * ({}) + 3. * ({})".format(
                diff_V.format(X="(X2 - 0.5 * X1)"),
                diff_V.format(X="(X2 + 0.5 * X1)"),
            ),

            # set the absorbing boundary
            abs_boundary="( "
                         "  sin(0.5 * pi * (X1 + X1_amplitude) / X1_amplitude) * "
                         "  sin(0.5 * pi * (X2 + X2_amplitude) / X2_amplitude) "
                         ") ** (0.05 * dt)",

            pi=np.pi,
        )

        ###################################################################
        #
        # Set the initial condition as given by Eq. (8)
        #
        ###################################################################
        R_bar = -30.
        sigma_R = 3.
        E_cm = 1.

        phi_rho = interparticle_sys.get_eigenstate(1).reshape(self.quant_sys.X1.shape)
        R = self.quant_sys.X2
        phi_R = np.exp(-(R - R_bar) ** 2 / (2. * sigma_R ** 2) + 1j * np.sqrt(4. * E_cm) * R)

        self.quant_sys.set_wavefunction(phi_rho * phi_R)

        ###################################################################
        #
        # Mask to calculate the tunnelling rate
        #
        ###################################################################
        rho = self.quant_sys.X1
        self.after_barrier = ((R + 0.5 * rho) > 4.)

        # List to save the transmission probability
        self.P_transmitted = []

    def __call__(self, frame_num):
        """
        Draw a new frame
        :param frame_num: current frame number
        :return: image objects
        """
        wavefunction = self.quant_sys.propagate(40)

        density = np.abs(wavefunction) ** 2

        # propagate and set the density
        self.img.set_array(density)

        self.P_transmitted.append(
            (self.quant_sys.t, density[self.after_barrier].sum() * self.quant_sys.dX1 * self.quant_sys.dX2)
        )

        # disable calculation of the Ehrenfest theorem
        self.quant_sys.isEhrenfest = False

        return self.img,

    def plot_potential_energy(self):
        """
        Axillary plotting facility:

        Plot the classically allowed and forbidden regions
        :return: None
        """
        from QuantumClassicalDynamics.wigner_normalize import WignerNormalize

        plt.imshow(
            ne.evaluate(self.quant_sys.V, local_dict=vars(self.quant_sys)),
            extent=self._extent,
            aspect=self._aspect,
            origin='lower',
            cmap='seismic',
            norm=WignerNormalize(vmiddle=self.quant_sys.hamiltonian_average[0]),
        )
        plt.xlabel('$R$ (a.u.)')
        plt.ylabel('$\\rho$ (a.u.)')
        plt.colorbar()
        plt.show()


fig = plt.gcf()
visualizer = VisualizeDynamics2D(fig)
animation = FuncAnimation(
    fig, visualizer, frames=np.arange(100), repeat=True, blit=True
)

plt.show()

# If you want to make a movie, comment "plt.show()" out and uncomment the lines bellow

# Set up formatting for the movie files
#   writer = writers['mencoder'](fps=10, metadata=dict(artist='a good student'), bitrate=-1)

# Save animation into the file
#   animation.save('2D_Schrodinger.mp4', writer=writer)

visualizer.plot_potential_energy()

# extract the reference to quantum system
quant_sys = visualizer.quant_sys

# Analyze how well the energy was preserved
h = np.array(quant_sys.hamiltonian_average)
print(
    "\nHamiltonian is preserved within the accuracy of {:.2e} percent".format(
        100. * (1. - h.min()/h.max())
    )
)

#################################################################
#
# Plot the Ehrenfest theorems after the animation is over
#
#################################################################

# generate time step grid
dt = quant_sys.dt
times = np.arange(dt, dt + dt*len(quant_sys.X1_average), dt)

plt.subplot(121)
plt.title("The first Ehrenfest theorem verification")

plt.plot(
    times,
    np.gradient(quant_sys.X1_average, dt),
    'r-',
    label='$d\\langle \\hat{x}_1 \\rangle/dt$'
)
plt.plot(
    times,
    quant_sys.X1_average_RHS,
    'b--', label='$\\langle \\hat{p}_1 \\rangle$'
)

plt.plot(
    times,
    np.gradient(quant_sys.X2_average, dt),
    'g-',
    label='$d\\langle \\hat{x}_2 \\rangle/dt$'
)
plt.plot(
    times,
    quant_sys.X2_average_RHS,
    'k--',
    label='$\\langle \\hat{p}_2 \\rangle$'
)

plt.legend()
plt.xlabel('time $t$ (a.u.)')

plt.subplot(122)
plt.title("The second Ehrenfest theorem verification")

plt.plot(
    times,
    np.gradient(quant_sys.P1_average, dt),
    'r-',
    label='$d\\langle \\hat{p}_1 \\rangle/dt$'
)
plt.plot(
    times,
    quant_sys.P1_average_RHS,
    'b--',
    label='$\\langle -\\partial\\hat{V}/\\partial\\hat{x}_1 \\rangle$'
)

plt.plot(
    times,
    np.gradient(quant_sys.P2_average, dt),
    'g-',
    label='$d\\langle \\hat{p}_2 \\rangle/dt$'
)
plt.plot(
    times,
    quant_sys.P2_average_RHS,
    'k--',
    label='$\\langle -\\partial\\hat{V}/\\partial\\hat{x}_2 \\rangle$'
)

plt.legend()
plt.xlabel('time $t$ (a.u.)')

plt.show()