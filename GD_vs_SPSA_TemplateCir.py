import pennylane as qml
from noisyopt import minimizeSPSA
import numpy as np
import matplotlib.pyplot as plt 
import time

def GD(steps=200, n_wires=4, n_layers=6, stepsize=0.3, device = qml.device("default.qubit", wires=4)):
    #USE pennylane's GradientDescentOptimizer PACKAGE TO CALCULATE ENERGIES

    all_pauliz_tensor_prod = qml.operation.Tensor(*[qml.PauliZ(i) for i in range(n_wires)])
    def circuit(params):
        qml.templates.StronglyEntanglingLayers(params, wires=list(range(n_wires)))
        return qml.expval(all_pauliz_tensor_prod)

    opt = qml.GradientDescentOptimizer(stepsize=stepsize)    
    cost = qml.QNode(circuit, device)

    params = qml.init.strong_ent_layers_normal(
        n_wires=n_wires, n_layers=n_layers
    )

    F = []
    for k in range(steps):
        params, energy = opt.step_and_cost(cost, params)
        F.append(energy)

    return F

def SPSA(steps=200, n_wires=4, n_layers=6, c=0.3, a=1.5, device = qml.device("default.qubit", wires=4)):
    #USE noisyopt's minimizeSPSA PACKAGE TO CALCULATE ENERGIES
    all_pauliz_tensor_prod = qml.operation.Tensor(*[qml.PauliZ(i) for i in range(n_wires)])
    def circuit(params):
        qml.templates.StronglyEntanglingLayers(params, wires=list(range(n_wires)))
        return qml.expval(all_pauliz_tensor_prod)

    qnode_spsa = qml.QNode(circuit, device)

    def cost_spsa(params):
        return qnode_spsa(params.reshape(n_layers, n_wires, 3))

    flat_shape = n_layers * n_wires * 3
    init_params = qml.init.strong_ent_layers_normal(
        n_wires=n_wires, n_layers=n_layers
    )
    init_params_spsa = init_params.reshape(flat_shape)

    cost_store_spsa = [cost_spsa(init_params_spsa)]
    device_execs_spsa = [0]


    F = []
    def callback_fn(xk):
        cost_val = cost_spsa(xk)
        F.append(cost_val)
        cost_store_spsa.append(cost_val)

    # Evaluate the initial cost
    cost_store_spsa = [cost_spsa(init_params_spsa)]

    res = minimizeSPSA(
        cost_spsa,
        x0=init_params_spsa.copy(),
        niter=steps,
        paired=False,
        c=c,
        a=a,
        callback=callback_fn,
    )

    return F

def plot_result(F, exact_E):
    Nitr=len(F)
    plt.figure()
    F = np.array(F)
    E = np.full((Nitr),exact_E)
    xv=np.arange(Nitr)+1
    plt.plot(xv, F)
    plt.plot(xv, E,'g--')
    plt.xlabel('iteration number')
    plt.ylabel('energy')
    plt.show()

if __name__ == '__main__':

    #write input prompt
    #pick steps, n_wire, & n_layers OR default params
    #hyper for spsa; c & a
    #hyper for gd; steps_size
    
    time_start_SPSA = time.time()
    print("Running SPSA...")
    res1 = SPSA()
    time_stop_SPSA = time.time()
    plot_result(res1, -1.136189454088)  

    time_start_GD = time.time()
    print("Running GD...")
    res2 = GD()
    time_stop_GD = time.time()
    plot_result(res2, -1.136189454088)


    print("SPSA time: " + str(time_stop_SPSA - time_start_SPSA))
    print("GD time: " + str(time_stop_GD - time_start_GD))

