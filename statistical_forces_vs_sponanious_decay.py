"""
Comparison of two different mechanism for tunneling enhancement:

    spontaneous decay vs statistical forces

Spontaneous decay is modeled by the translationary invariant master equations [https://arxiv.org/abs/1706.00341]
"""
from QuantumClassicalDynamics.wavefunc_monte_carlo1D import *
from multiprocessing import Pool


def propagate_traj(args):
    """
    A function that propagates a single quantum trajectory
    :param params: dictionary of parameters to initialize the Monte Carlo propagator
    :param seed: the seed for random number generator
    :return: numpy.array contanning the final wavefunction
    """
    # extract the propagator parameters and seed
    params, seed = args

    # Since there are many trajectories are run in parallel use only a single thread
    ne.set_num_threads(1)

    # Set the seed for random number generation to avoid the artifact described in
    #   https://stackoverflow.com/questions/24345637/why-doesnt-numpy-random-and-multiprocessing-play-nice
    # It is recommended that seeds be generate via the function get_seeds (see below)
    np.random.seed(seed)

    # initialize the propagator
    qsys = WavefuncMonteCarloPoission(**params)
    qsys.set_wavefunction(qsys.initial_condition)

    # propagate
    return qsys.propagate(qsys.ntsteps)


def get_seeds(size):
    """
    Generate unique random seeds for subsequently seeding them into random number generators in multiprocessing simulations

    This utility is to avoid the following artifact:
        https://stackoverflow.com/questions/24345637/why-doesnt-numpy-random-and-multiprocessing-play-nice
    :param size: number of samples to generate
    :return: numpy.array of np.uint32
    """
    # Note that np.random.seed accepts 32 bit unsigned integers

    # get the maximum value of np.uint32 can take
    max_val = np.iinfo(np.uint32).max

    # A set of unique and random np.uint32
    seeds = set()

    # generate random numbers until we have sufficiently many nonrepeating numbers
    while len(seeds) < size:
        seeds.update(
            np.random.randint(max_val, size=size, dtype=np.uint32)
        )

    # make sure we do not return more numbers that we are asked for
    return np.fromiter(seeds, np.uint32, size)


def get_rho_monte_carlo(params):
    """
    Obtain the density matrix via the wave function Monte Carlo propagation
    :param params: (dict) to initialize the monte carlo propagator
    :return: numpy.array containing the density matrix
    """
    X_gridDIM = params["X_gridDIM"]

    # allocate memory for additional matrices
    monte_carlo_rho = np.zeros((X_gridDIM, X_gridDIM), dtype=np.complex)
    rho_traj = np.zeros_like(monte_carlo_rho) # monte_carlo_rho is average of rho_traj

    # the iterator to launch (12 * chunksize) trajectories
    chunksize = 1000
    iter_trajs = ((params, seed) for seed in get_seeds(12 * chunksize))

    # run each Monte Carlo trajectories on multiple cores
    with Pool() as pool:
        # index counting the trajectory needed to calculate the mean iteratively
        t = 0
        for psi in pool.imap_unordered(propagate_traj, iter_trajs, chunksize=chunksize):

            # form the density matrix out of the wavefunctions
            np.outer(psi.conj(), psi, out=rho_traj)

            # Calculate the iterative mean following http://www.heikohoffmann.de/htmlthesis/node134.html
            rho_traj -= monte_carlo_rho
            rho_traj /= (t + 1)
            monte_carlo_rho += rho_traj

            # increment the trajectory counter
            t += 1

    return monte_carlo_rho


if __name__ =='__main__':

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    ##############################################################################################
    #
    # Setup parameters of the tunneling system
    #
    ##############################################################################################

    params = dict(
        t=0.,
        dt=0.002,
        X_gridDIM=1024,
        X_amplitude=30.,

        # Kinetic energy
        K="0.5 * P ** 2",
        diff_K="P",


        # Potential energy
        U0=4.,
        V="U0 * exp(-0.5 * X ** 2)",
        diff_V="-X * U0 * exp(-0.5 * X ** 2)",

        # use absorbing boundary
        abs_boundary="sin(0.5 * pi * (X + X_amplitude) / X_amplitude) ** (0.005 * dt)",
        pi=np.pi,

        # number of steps to propagate
        ntsteps=2800,

        # initial wave function
        initial_condition="exp(-1. * (X + 4.5) ** 2 + 1j * X)",
    )

    # initialize the propagator for coherent tunneling
    coherent_tunneling = SplitOpSchrodinger1D(**params)
    coherent_tunneling.set_wavefunction(coherent_tunneling.initial_condition)

    ##############################################################################################
    #
    # Setup spontaneous emission propagator
    #
    #   see Eq. (13) of https://arxiv.org/abs/1706.00341
    #
    ##############################################################################################

    def apply_spont_emission_plus(self):
        """
        Apply
            A^+  = exp(-1j * p0 * X) * C * exp(q * P)
        onto the wavefunction
        :param self:
        :return: None
        """
        # Go to the momentum representation
        ne.evaluate("(-1) ** k * wavefunction", local_dict=vars(self), out=self.wavefunction)
        self.wavefunction = fftpack.fft(self.wavefunction, overwrite_x=True)

        ne.evaluate("exp(q * P) * wavefunction", local_dict=vars(self), out=self.wavefunction)

        # Go back to the coordinate representation
        self.wavefunction = fftpack.ifft(self.wavefunction, overwrite_x=True)
        ne.evaluate("exp(-1j * p0 * X) * (-1) ** k * wavefunction", local_dict=vars(self), out=self.wavefunction)

    def apply_spont_emission_minus(self):
        """
         Apply
            A^-  = exp(1j * p0 * X) * C * exp(-q * P)
        onto the wavefunction
        :param self:
        :return: None
        """
        # Go to the momentum representation
        ne.evaluate("(-1) ** k * wavefunction", local_dict=vars(self), out=self.wavefunction)
        self.wavefunction = fftpack.fft(self.wavefunction, overwrite_x=True)

        ne.evaluate("exp(-q * P) * wavefunction", local_dict=vars(self), out=self.wavefunction)

        # Go back to the coordinate representation
        self.wavefunction = fftpack.ifft(self.wavefunction, overwrite_x=True)
        ne.evaluate("exp(1j * p0 * X) * (-1) ** k * wavefunction", local_dict=vars(self), out=self.wavefunction)

    spontaneous_emission_params = params.copy()
    spontaneous_emission_params.update(
        C=20.,
        p0=0.1,
        q=0.008,

        BdaggerB_P=("(C * exp(q * P)) ** 2", "(C * exp(-q * P)) ** 2"),
        apply_B=(apply_spont_emission_plus, apply_spont_emission_minus)
    )

    ##############################################################################################
    #
    # Setup the statistical force that compensates the barrier
    #
    #   see Eq. (6a) of https://arxiv.org/abs/1611.02736
    #
    ##############################################################################################

    stat_force_params = params.copy()

    C = 25.
    p0 = 0.001

    # Magnetic field [Eq. (9)]
    diff_V = ne.evaluate(stat_force_params["diff_V"], local_dict=vars(coherent_tunneling))

    B = 1. / (16 * C ** 4) * diff_V * np.sqrt(16 * p0 ** 2 * C ** 4 - diff_V ** 2)

    ((16 * p0 ** 2 * C ** 4 - diff_V ** 2).min())

    #assert any(np.isnan(B)), "adjust p0 and C"

    def apply_A_plus(self):
        self.wavefunction *= self.A_plus

    def apply_A_minus(self):
        self.wavefunction *= self.A_minus

    stat_force_params.update(
        A_plus=C * np.exp(
            2j * np.cumsum(np.sqrt(p0 ** 2 + 2 * B)) * coherent_tunneling.dX
        ),

        A_minus=C * np.exp(
            -2j * np.cumsum(np.sqrt(p0 ** 2 - 2 * B)) * coherent_tunneling.dX
        ),

        C = C,

        AdaggerA_X=("C ** 2", "C ** 2"),
        apply_A=(apply_A_plus, apply_A_minus),
    )

    ##############################################################################################
    #
    #   Propagate
    #
    ##############################################################################################

    coherent_tunneling.propagate(coherent_tunneling.ntsteps)

    spontaneous_emission_rho = get_rho_monte_carlo(spontaneous_emission_params)

    stat_force_rho = get_rho_monte_carlo(stat_force_params)

    def get_purity(rho, dX=coherent_tunneling.dX):
        """
        Calculate purity
        :param rho: numpy.array saving the density matrix
        :param dX: (float) coordinate step size
        :return: float
        """
        return np.sum(np.abs(rho) ** 2) * dX ** 2

    print(
        "Purity of spontaneous emission: {:.3f}".format(
            get_purity(spontaneous_emission_rho)
        )
    )
    print(
        "Purity of environmentally assisted tunneling: {:.3f}".format(
            get_purity(stat_force_rho)
        )
    )

    ##############################################################################################
    #
    #   Save the propagation results
    #
    ##############################################################################################

    import pickle
    with open("RESULTS.pickle", 'bw') as f:
        pickle.dump(
            {
                "coherent_tunneling" : coherent_tunneling,
                "spontaneous_emission_rho" : spontaneous_emission_rho,
                "stat_force_rho" : stat_force_rho,
            },
            f
        )

    ##############################################################################################
    #
    #   Plot: Compare the final states
    #
    ##############################################################################################

    plt.title("Coordinate probability density")
    plt.semilogy(
        coherent_tunneling.X,
        spontaneous_emission_rho.diagonal(),
        label="Spontaneous emission"
    )
    plt.semilogy(
        coherent_tunneling.X,
        stat_force_rho.diagonal(),
        label="environmentally assisted tunneling"
    )
    plt.semilogy(
        coherent_tunneling.X,
        np.abs(coherent_tunneling.wavefunction) ** 2,
        label="Coherent tunneling"
    )
    plt.xlabel("$x$ (a.u.)")
    plt.ylabel("Coordinate probability density")
    plt.ylim((1e-7, 1.))
    plt.legend()
    plt.show()

    ##############################################################################################
    #
    #   Compare individual coherent and incoherent evolutions
    #
    ##############################################################################################

    """
    plt.title("Plot the potential barrier")
    plt.plot(
        propagator.X,
        ne.evaluate(propagator.V, local_dict=vars(propagator)),
        '-*'
    )
    plt.show()
    """

    """
    def analyze_single_propagation(propagator):
        indx = np.where(propagator.X > 2)

        densities = []
        tunnelling_probability = []

        # set the initial condition
        propagator.set_wavefunction(propagator.initial_condition)

        for _ in range(propagator.ntsteps):
            density = np.abs(propagator.propagate()) ** 2

            densities.append(density)
            tunnelling_probability.append(
                np.sum(density[indx]) * propagator.dX
            )
        return densities, tunnelling_probability

    coherent_tunneling = SplitOpSchrodinger1D(**params)
    coherent_density, coherent_tunnelling = analyze_single_propagation(coherent_tunneling)

    spontaneous_emission_density, spontaneous_emission_tunnelling = analyze_single_propagation(
        WavefuncMonteCarloPoission(**spontaneous_emission_params)
    )

    plt.subplot(121)
    plt.title("Coherenet evolution: Probability density")
    img_params = dict(
        origin = 'lower',
        norm = LogNorm(1e-12, 1.),
        #cmap = 'jet',
    )
    plt.imshow(coherent_density, **img_params)

    plt.subplot(122)
    plt.title("Spontaneous evolution: Probability density")
    plt.imshow(spontaneous_emission_density, **img_params)

    plt.show()

    plt.title("Final state")
    plt.semilogy(
        coherent_tunneling.X,
        spontaneous_emission_density[-1],
        label="Spontaneous emission"
    )
    plt.semilogy(
        coherent_tunneling.X,
        coherent_density[-1],
        label="Coherent tunneling"
    )
    plt.xlabel("$x$ (a.u.)")
    plt.ylabel("Probability")
    plt.ylim((1e-4, 1.))
    plt.legend()
    plt.show()

    plt.title("Tunneling probability")
    plt.plot(coherent_tunnelling, label="Coherent tunneling")
    plt.plot(spontaneous_emission_tunnelling, label="Spontaneous emission")
    plt.legend()
    plt.show()
    """

    ##################################################################################################

    """
    plt.subplot(131)
    plt.title("Verify the first Ehrenfest theorem")

    times = coherent_tunneling.dt * np.arange(len(coherent_tunneling.X_average))
    plt.plot(
        times,
        np.gradient(coherent_tunneling.X_average, coherent_tunneling.dt),
        '-r',
        label='$d\\langle\\hat{x}\\rangle / dt$'
    )
    plt.plot(times, coherent_tunneling.X_average_RHS, '--b', label='$\\langle\\hat{p}\\rangle$')
    plt.legend()
    plt.ylabel('momentum')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(132)
    plt.title("Verify the second Ehrenfest theorem")

    plt.plot(
        times,
        np.gradient(coherent_tunneling.P_average, coherent_tunneling.dt),
        '-r',
        label='$d\\langle\\hat{p}\\rangle / dt$'
    )
    plt.plot(times, coherent_tunneling.P_average_RHS, '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
    plt.legend()
    plt.ylabel('force')
    plt.xlabel('time $t$ (a.u.)')

    plt.subplot(133)
    plt.title("The expectation value of the hamiltonian")

    # Analyze how well the energy was preserved
    h = np.array(coherent_tunneling.hamiltonian_average)
    print(
        "\nHamiltonian is preserved within the accuracy of {:2e} percent".format(
            100. * (1. - h.min() / h.max())
        )
    )

    plt.plot(times, h)
    plt.ylabel('energy')
    plt.xlabel('time $t$ (a.u.)')

    plt.show()
    """
