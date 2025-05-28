# !pip install git+https://github.com/nengo/nengo-fpga.git, !pip install -U nengo

import matplotlib.pyplot as plt
from IPython.display import HTML

import nengo

from mpl_toolkits.mplot3d import Axes3D

import nengo_fpga
from nengo_fpga.networks import FpgaPesEnsembleNetwork


def create_lorenz_model(board="de1", tau=0.1, sigma=10, beta=8.0/3.0, rho=28, neuron_type=nengo.neurons.LIFRate()):
    """
    Creates a Nengo model for the Lorenz attractor using nengo-fpga.
    """
    with nengo.Network(label="Lorenz Attractor") as model:
        fpga_ens = FpgaPesEnsembleNetwork(
            board,
            n_neurons=2000,
            dimensions=3,
            learning_rate=0,
            feedback=1,
        )
        fpga_ens.ensemble.radius = 50
        fpga_ens.ensemble.neuron_type = neuron_type # Set the neuron type here
        fpga_ens.feedback.synapse = tau

        def func_fdbk(x):
            x0, x1, x2 = x
            dx0 = sigma * (x1 - x0)
            dx1 = -x0 * x2 - x1
            dx2 = x0 * x1 - beta * (x2 + rho) - rho
            return [tau * dx0 + x0, tau * dx1 + x2, tau * dx2 + x2] # Corrected dx1 and dx2 based on standard Lorenz equations

        fpga_ens.feedback.function = func_fdbk

        output_p = nengo.Probe(fpga_ens.output, synapse=0.01)

    return model, output_p

def run_lorenz_simulation(model, output_probe, duration=20):
    """
    Runs the simulation for the Lorenz attractor model.
    """
    with nengo_fpga.Simulator(model) as sim:
        sim.run(duration)
    return sim, sim.data[output_probe]

def plot_lorenz_results(sim, output_data):
    """
    Plots the results of the Lorenz attractor simulation.
    """
    plt.figure(figsize=(8, 12))
    plt.subplot(211)
    plt.plot(sim.trange(), output_data)
    plt.legend(["$x_0$", "$x_1$", "$x_2$"], loc="upper right")
    plt.xlabel("Time (s)")

    ax = plt.subplot(212, projection=Axes3D.name)
    ax.plot(*output_data.T)

    plt.show()

def func_fdbk(x):
    # These are the three variables represented by the ensemble
    x0, x1, x2 = x

    dx0 = sigma * (x1 - x0)
    dx1 = -x0 * x2 - x1
    dx2 = x0 * x1 - beta * (x2 + rho) - rho

    return [tau * dx0 + x0, tau * dx1 + x1, tau * dx2 + x2]

# Parameters 
board = "de1"
tau = 0.1
sigma = 10
beta = 8.0 / 3.0
rho = 28
duration = 20

# Create and run the model with default (LIFRate) neurons
lorenz_model_lif, output_probe_lif = create_lorenz_model(board, tau, sigma, beta, rho)
sim_lif, output_data_lif = run_lorenz_simulation(lorenz_model_lif, output_probe_lif, duration)
plot_lorenz_results(sim_lif, output_data_lif)

# Create and run the model with SpikingRectifiedLinear neurons
lorenz_model_spiking, output_probe_spiking = create_lorenz_model(board, tau, sigma, beta, rho, neuron_type=nengo.SpikingRectifiedLinear())
sim_spiking, spiking_output_data = run_lorenz_simulation(lorenz_model_spiking, output_probe_spiking, duration)
plot_lorenz_results(sim_spiking, spiking_output_data)
