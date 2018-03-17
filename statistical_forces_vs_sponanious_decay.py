"""
Comparison of two different mechanism for tunneling enhancement:

    spontaneous decay vs statistical forces

Spontaneous decay is modeled by the translationary invariant master equations [https://arxiv.org/abs/1706.00341]
"""
import QuantumClassicalDynamics.wavefunc_monte_carlo1D
from QuantumClassicalDynamics.wavefunc_monte_carlo1D import WavefuncMonteCarloPoission, SplitOpSchrodinger1D, np
from QuantumClassicalDynamics.split_op_schrodinger1D import SplitOpSchrodinger1D, np, ne

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