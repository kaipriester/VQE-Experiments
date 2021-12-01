import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt 
import time

dev = qml.device('default.qubit', wires=3, shots=None)

init_param = [
    np.array([0.3, 0.2, 0.67]),
    np.array(1.1),
    np.array([-0.2, 0.1, -2.5]),
]

num_freqs = [[1, 1, 1], 3, [2, 2, 2]]

@qml.qnode(dev)
def cost_function(rot_param, layer_par, crot_param):
    for i, par in enumerate(rot_param):
        qml.RX(par, wires=i)

    for w in dev.wires:
        qml.RX(layer_par, wires=w)

    for i, par in enumerate(crot_param):
        qml.CRY(par, wires=[i, (i+1)%3])

    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

#TRY ROTOSOLVE AND ROTOSELECT
def RotoSolve(num_steps, init_params):
    opt = qml.optimize.RotosolveOptimizer()
    param = init_params.copy()
    cost_rotosolve  = []

    for step in range(num_steps):
        param, cost, sub_cost = opt.step_and_cost(
            cost_function,
            *param,
            num_freqs=num_freqs,
            full_output=True,
        )
        print(f"Cost before step: {cost}")
        print(f"Minimization substeps: {np.round(sub_cost, 6)}")
        cost_rotosolve.extend(sub_cost)

    return cost_rotosolve    

def ShotAdaptive(num_step):
    coeffs = [2, 4, -1, 5, 2]
    obs = [
        qml.PauliX(1),
        qml.PauliZ(1),
        qml.PauliX(0) @ qml.PauliX(1),
        qml.PauliY(0) @ qml.PauliY(1),
        qml.PauliZ(0) @ qml.PauliZ(1)
    ]
    H = qml.Hamiltonian(coeffs, obs)
    dev = qml.device("default.qubit", wires=2, shots=100)
    cost = qml.ExpvalCost(qml.templates.StronglyEntanglingLayers, H, dev)  

    shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
    params = np.random.random(shape)
    opt = qml.ShotAdaptiveOptimizer(min_shots=10)

    energy = []
    for i in range(num_step):
       params = opt.step(cost, params)
       energy = cost (params)
       print(f"Step {i}: cost = {energy:.2f}, shots_used = {opt.total_shots_used}")

    return energy   

def plot_result(F, exact_E=-0.1):
    Nitr=len(F)
    fig = plt.figure()
    F = np.array(F)
    #E = np.full((Nitr),exact_E)
    xv=np.arange(Nitr)+1
    plt.plot(xv, F)
    #plt.plot(xv, E,'g--')
    plt.xlabel('iteration number')
    plt.ylabel('energy')
    plt.show()   

if __name__ == '__main__':
    RotoSolve(50, init_params=init_param)
    energy = ShotAdaptive(50)
    plot_result(energy)
