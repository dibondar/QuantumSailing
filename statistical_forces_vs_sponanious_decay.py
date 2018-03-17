"""
Comparison of two different mechanism for tunneling enhancement:

    spontaneous decay vs statistical forces

Spontaneous decay is modeled by the translationary invariant master equations [https://arxiv.org/abs/1706.00341]
"""
from QuantumClassicalDynamics.wavefunc_monte_carlo1D import *
from multiprocessing import Pool


def propagate_traj(params):
    """
    A function that propagates a single quantum trajectory
    :param params: dictionary of parameters to initialize the Monte Carlo propagator
    :return: numpy.array contanning the final wavefunction
    """
    # Since there are many trajectories are run in parallel use only a single thread
    ne.set_num_threads(1)

    # initialize the propagator
    qsys = WavefuncMonteCarloPoission(**params)
    qsys.set_wavefunction(qsys.initial_condition)

    # propagate
    return qsys.propagate(qsys.ntsteps)


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

    # launch 1000 trajectories
    from itertools import repeat
    chunksize = 10
    trajs = repeat(params, 12 * chunksize)

    # run each Monte Carlo trajectories on multiple cores
    with Pool() as pool:
        # index counting the trajectory needed to calculate the mean iteratively
        t = 0
        for psi in pool.imap_unordered(propagate_traj,  trajs, chunksize=chunksize):

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
        X_amplitude=24.,

        # Kinetic energy
        K="0.5 * P ** 2",
        diff_K="P",


        # Potential energy
        U0=4.,
        V="U0 * exp(-0.5 * X ** 2)",
        diff_V="-X * U0 * exp(-0.5 * X ** 2)",

        # use absorbing boundary
        abs_boundary="sin(0.5 * pi * (X + X_amplitude) / X_amplitude) ** (0.02 * dt)",
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
    ##############################################################################################

    def apply_spont_emission_pluse(self):
        """
        Apply
            A = exp(-1j * p0 * X) * C * exp(k * P)
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
        The complex conjugate version of the previous function
        :param self:
        :return: None
        """
        # Go to the momentum representation
        ne.evaluate("(-1) ** k * wavefunction", local_dict=vars(self), out=self.wavefunction)
        self.wavefunction = fftpack.fft(self.wavefunction, overwrite_x=True)

        ne.evaluate("exp(q * P) * wavefunction", local_dict=vars(self), out=self.wavefunction)

        # Go back to the coordinate representation
        self.wavefunction = fftpack.ifft(self.wavefunction, overwrite_x=True)
        ne.evaluate("exp(1j * p0 * X) * (-1) ** k * wavefunction", local_dict=vars(self), out=self.wavefunction)

    spontaneous_emission_params = params.copy()
    spontaneous_emission_params.update(
        C=4.,
        p0=0.01,
        q=0.01,

        BdaggerB_P=("(C * exp(q * P)) ** 2", "(C * exp(q * P)) ** 2"),
        apply_B=(apply_spont_emission_pluse, apply_spont_emission_minus)
    )

    ##############################################################################################
    #
    #   Propagate
    #
    ##############################################################################################

    coherent_tunneling.propagate(coherent_tunneling.ntsteps)

    spontaneous_emission_rho = get_rho_monte_carlo(spontaneous_emission_params)

    ##############################################################################################
    #
    #   Plot
    #
    ##############################################################################################

    plt.title("")
    plt.semilogy(
        coherent_tunneling.X,
        spontaneous_emission_rho.diagonal(),
        label="Spontaneous emission"
    )
    plt.semilogy(
        coherent_tunneling.X,
        np.abs(coherent_tunneling.wavefunction) ** 2,
        label="Coherent tunneling"
    )
    plt.xlabel("$x$ (a.u.)")
    plt.ylabel("Probability")
    plt.ylim((1e-4, 1.))
    plt.legend()
    plt.show()


    """
    plt.title("Plot the potential barrier")
    plt.plot(
        coherent_tunneling.X,
        ne.evaluate(coherent_tunneling.V, local_dict=vars(coherent_tunneling)),
        '-*'
    )
    plt.show()
    

    plt.title("Probability density")
    plt.imshow(
        [np.abs(coherent_tunneling.propagate()) ** 2 for _ in range(coherent_tunneling.ntsteps)],
        origin='lower',
        norm=LogNorm(1e-12, 1.),
        cmap='jet',
    )
    plt.show()
    

    indx = np.where(coherent_tunneling.X > 2)
    tunnelling_probability = [
        np.sum(np.abs(coherent_tunneling.propagate()[indx]) ** 2) * coherent_tunneling.dX
        for _ in range(coherent_tunneling.ntsteps)
    ]
    plt.plot(tunnelling_probability)
    plt.show()
    """

    """
    ##################################################################################################

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