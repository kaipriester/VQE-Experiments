import pennylane as qml
from noisyopt import minimizeSPSA
import numpy as np
import matplotlib.pyplot as plt 
import time

from pennyl3 import make_circuit, sim_run

def GD(steps=100, stepsize=0.3, init_params = np.pi/2*np.ones(6)):
    opt = qml.GradientDescentOptimizer(stepsize=stepsize)    
    cost = sim_run

    params = init_params

    F = []
    for k in range(steps):
        params, energy = opt.step_and_cost(cost, params)
        F.append(energy)
        print("energy at step " + str(k) +": " + str(energy)) 

    return F   

def SPSA(steps=100, c=0.2, a=1.5, init_params = np.pi/2*np.ones(6)):
    init_params_spsa = init_params

    F = []
    def callback_fn(xk):
        cost_val = sim_run(xk)
        F.append(cost_val)
        cost_store_spsa.append(cost_val)

    cost_store_spsa = [sim_run(init_params_spsa)]

    res = minimizeSPSA(
        sim_run,
        x0=init_params_spsa.copy(),
        niter=steps,
        paired=False,
        c=c,
        a=a,
        callback=callback_fn,)

    return F

def Adam(steps=100, stepsize=0.3, init_params = np.pi/2*np.ones(6)):
    opt = qml.AdamOptimizer(stepsize=stepsize)    
    cost = sim_run

    params = init_params

    F = []
    for k in range(steps):
        params, energy = opt.step_and_cost(cost, params)
        F.append(energy)
        print("energy at step " + str(k) +": " + str(energy)) 

    return F    

def plot_result(F, exact_E=-0.1):
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
    print("In this program you can analyze the performance of different QML optimization methods on the quamtum circuit defined in pennyl3.py.")
    valid_input = False

    while not valid_input:
        print("Menu: ")
        print("     1. Basic gradient-descent optimizer")
        print("     2. Simultaneous Perturbation Stochastic Approximation (SPSA)")
        print("     3. Gradient-descent optimizer with adaptive learning rate, first and second moment (AdamOptimizer)")
        print("Input #: ", end='')
        menu_input = input()

        if(menu_input == "1"):
            valid_input = True

            time_start_GD = time.time()
            print("Running GD-----------------------------------------------------------------------------------------------")
            res2 = GD()
            time_stop_GD = time.time()
            plot_result(res2)

            print("GD time " + str(time_stop_GD - time_start_GD))

        elif(menu_input == "2"):
            valid_input = True
            time_start_SPSA = time.time()
            print("Running SPSA---------------------------------------------------------------------------------------------")
            res1 = SPSA()
            time_stop_SPSA = time.time()
            plot_result(res1)  

            print("SPSA time: " + str(time_stop_SPSA - time_start_SPSA))

        elif(menu_input == "3"):
            valid_input = True

            time_start_Adam = time.time()
            print("Running Adam---------------------------------------------------------------------------------------------")
            res3 = Adam()
            time_stop_Adam = time.time()
            plot_result(res3)

            print("Adam time: " + str(time_stop_Adam - time_start_Adam))

        else:
            valid_input = False
            print("Invalid Input!")
            print()    
    
    
    
    
 