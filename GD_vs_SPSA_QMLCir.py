import pennylane as qml
from noisyopt import minimizeSPSA
import numpy as np
import matplotlib.pyplot as plt 
import time


def my_ansatz(p,w0,w1):
    qml.CNOT(wires=[w1,w0])
    qml.RY(p,wires=w1)
    qml.RZ(np.pi,wires=w1)
    qml.CNOT(wires=[w0,w1])
    qml.RZ(-np.pi,wires=w1)
    qml.RY(-p,wires=w1)
    qml.CNOT(wires=[w1,w0])
        
def my_circuit(params):
    qml.PauliX(wires=1)
    qml.PauliX(wires=2)
    my_ansatz(params[0],0,1)
    my_ansatz(params[1],2,3)
    my_ansatz(np.pi/2,1,2)
    my_ansatz(params[2],0,1)
    my_ansatz(params[3],2,3)
    my_ansatz(np.pi/2,1,2)
    my_ansatz(params[4],0,1)
    my_ansatz(params[5],2,3)
        
#@qml.qnode(device)
def circuit(param, n=3, measurenum=0):
    # measurement circuit
    my_circuit(param)
    
    measurestr=''
    if measurenum==0: 
        measurestr='ZZZZ'
    elif measurenum==1: 
        measurestr='XXXX'
    elif measurenum==2: 
        measurestr='YYYY'
            
        
    for qb in range(n):
        if (measurestr[qb] == 'X'):  # Hadamard transform between X and Z
            qml.Hadamard(wires=qb)
        if (measurestr[qb] == 'Y'):  # transform
            qml.S(wires=qb)
            qml.Hadamard(wires=qb)
         
    return qml.probs(wires=[0,1,2,3]) 
        #return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

def GD(steps=10, n_wires=4, n_layers=6, stepsize=0.3, exact_E = -0.1026, device = qml.device("default.qubit", wires = 4, shots = 8000)):
    #USE pennylane's GradientDescentOptimizer PACKAGE TO CALCULATE ENERGIES

    opt = qml.GradientDescentOptimizer(stepsize=stepsize)    
    cost = qml.QNode(circuit, device)

    params = qml.init.strong_ent_layers_normal(
        n_wires=n_wires, n_layers=n_layers
    )
    params = np.zeros(n_layers)

    F = []
    for k in range(steps):
        params, energy = opt.step_and_cost(cost, params)
        F.append(energy)
        print("energy at step " + str(k) +": " + str(energy)) 

    return F   

def SPSA(steps=10, n_wires=4, n_layers=6, c=0.3, a=1.5, exact_E = -0.1026, device = qml.device("default.qubit", wires = 4, shots = 8000)):
  #USE noisyopt's minimizeSPSA PACKAGE TO CALCULATE ENERGIES
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


    #print("SPSA time: " + str(time_stop_SPSA - time_start_SPSA))
    print("GD time: " + str(time_stop_GD - time_start_GD))
 