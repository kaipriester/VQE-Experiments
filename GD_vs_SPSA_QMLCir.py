import pennylane as qml
from noisyopt import minimizeSPSA
import numpy as np
import matplotlib.pyplot as plt 
import time

def sigma(first, last, const):
    # first : is the first value of (n) (the index of summation)
    # last : is the last value of (n)
    # const : is the number that you want to sum its multiplication each (n) times with (n)
    sum = 0
    for i in range(first, last + 1):
        sum += const * i
    return sum

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
def make_circuit(param, measurenum=0):
    # measurement circuit
    my_circuit(param)
    
    measurestr=''
    if measurenum==0: 
        measurestr='ZZZZ'
    elif measurenum==1: 
        measurestr='XXXX'
    elif measurenum==2: 
        measurestr='YYYY'
            
    n = 4    
    for qb in range(n):
        if (measurestr[qb] == 'X'):  # Hadamard transform between X and Z
            qml.Hadamard(wires=qb)
        if (measurestr[qb] == 'Y'):  # transform
            qml.S(wires=qb)
            qml.Hadamard(wires=qb)

    return qml.probs(wires=[0,1,2,3]) 
    # return [
    #     qml.expval(qml.PauliX(0) @ qml.PauliY(1)),
    #     qml.expval(qml.PauliX(0) @ qml.PauliZ(2))
    # ]

def GD(steps=10, n_wires=4, n_layers=6, stepsize=0.3, device = qml.device("default.qubit", wires = 4, shots = 8000)):
    #USE pennylane's GradientDescentOptimizer PACKAGE TO CALCULATE ENERGIES

    opt = qml.GradientDescentOptimizer(stepsize=stepsize)    
    cost = qml.QNode(make_circuit, device)

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

def SPSA(steps=200, n_wires=4, n_layers=6, c=0.3, a=1.5, device = qml.device("default.qubit", wires = 4, shots = 8000)):
  #USE noisyopt's minimizeSPSA PACKAGE TO CALCULATE ENERGIES
    qnode_spsa = qml.QNode(make_circuit, device)

    def cost_spsa(params):
        nda = qnode_spsa(params)
        sum_a=0
        A=0.1*np.ones(16) # weights, 16 for total 16 states, its value can be change​
        for i,j in enumerate(nda):
            sum_a=sum_a+A[i]*j
        print(nda)
        print(sum_a)    
        return sum_a

    init_params_spsa = [0,0,0,0,0,0]

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

def plot_result(F, exact_E=0):
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
    plot_result(res1)  

    time_start_GD = time.time()
    print("Running GD...")
    res2 = GD()
    time_stop_GD = time.time()
    plot_result(res2)


    print("SPSA time: " + str(time_stop_SPSA - time_start_SPSA))
    print("GD time: " + str(time_stop_GD - time_start_GD))
 