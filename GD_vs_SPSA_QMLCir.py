import pennylane as qml
from noisyopt import minimizeSPSA
import numpy as np
import matplotlib.pyplot as plt 
import time

from pennyl3 import sim_run

def GD(steps, init_params, stepsize):
    opt = qml.GradientDescentOptimizer(stepsize=stepsize)    
    cost = sim_run

    params = init_params

    F = []
    for k in range(steps):
        params, energy = opt.step_and_cost(cost, params)
        F.append(energy)
        print("energy at step " + str(k) +": " + str(energy)) 

    return F   

def SPSA(steps, init_params, c, a):
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

def Adam(steps, init_params, stepsize):
    opt = qml.AdamOptimizer(stepsize=stepsize)    
    cost = sim_run

    params = init_params

    F = []
    for k in range(steps):
        params, energy = opt.step_and_cost(cost, params)
        F.append(energy)
        print("energy at step " + str(k) +": " + str(energy)) 

    return F    

def plot_result(F, opt_name, exact_E=-0.1):
    Nitr=len(F)
    fig = plt.figure()
    F = np.array(F)
    E = np.full((Nitr),exact_E)
    xv=np.arange(Nitr)+1
    plt.plot(xv, F)
    plt.plot(xv, E,'g--')
    plt.xlabel('iteration number')
    plt.ylabel('energy')
    plt.savefig('results/' + opt_name + '_plot.png')
    plt.show()

    return fig

if __name__ == '__main__':
    print("In this program you can analyze the performance of different QML optimization methods on the quamtum circuit defined in pennyl3.py.")
    valid_input = False

    while not valid_input:
        print("     Menu: ")
        print("     1. Basic gradient-descent optimizer (GD)")
        print("     2. Simultaneous Perturbation Stochastic Approximation (SPSA)")
        print("     3. Gradient-descent optimizer with adaptive learning rate, first and second moment (Adam)")
        print("Input menu ID(s) of optimizers to run (examples: 1,3 or 2): ", end='')
        menu_input = input()   
        menu_input = list(menu_input.split(","))

        for i in menu_input:
            if(i == '1'):
                valid_input = True
                print("GD hyperparameters (input d to use defaults OR any key to continue modifying): ", end='')
                param_spec = input()
                if(param_spec != "d"):
                    print("Number of iterations (example: 300): ", end='')
                    niter = int(input())
                    print("Initial parameters (example: 1.57079633,1.57079633,1.57079633,1.57079633,1.57079633,1.57079633): ", end='')
                    init_params = input()
                    init_params = [float(item) for item in init_params.split(',')]
                    print("Stepsize (example 0.2): ", end='')
                    step_size = float(input())
                else: 
                    niter = 100
                    init_params = np.pi/2*np.ones(6)
                    step_size = 0.3    
                
                time_start = time.time()
                print("Running GD-----------------------------------------------------------------------------------------------")
                result = GD(niter, init_params, step_size)
                time_stop = time.time()
                print("GD plot saved to results directory")
                print("GD time elapsed: " + str(time_stop - time_start))
                print()
                fig = plot_result(result, "GD")

            elif(i == '2'):
                valid_input = True
                print("SPSA hyperparameters (input d to use defaults OR any key to continue modifying): ", end='')
                param_spec = input()
                if(param_spec != "d"):
                    print("Number of iterations (example: 300): ", end='')
                    niter = int(input())
                    print("Initial parameters (example: 1.57079633,1.57079633,1.57079633,1.57079633,1.57079633,1.57079633): ", end='')
                    init_params = input()
                    init_params = [float(item) for item in init_params.split(',')]
                    print("c (example 0.2): ", end='')
                    c = float(input())
                    print("a (example 1.5): ", end='')
                    a = float(input())

                else: 
                    niter = 100
                    init_params = np.pi/2*np.ones(6)
                    c = 0.2
                    a = 1.5  

                time_start = time.time()
                print("Running SPSA---------------------------------------------------------------------------------------------")
                result = SPSA(niter, init_params, c, a)
                time_stop = time.time()
                print("SPSA plot saved to results directory")
                print("SPSA time elapsed: " + str(time_stop - time_start))
                print()
                fig = plot_result(result, "SPSA")

            elif(i == '3'):
                valid_input = True
                print("Adam hyperparameters (input d to use defaults OR any key to continue modifying): ", end='')
                param_spec = input()
                if(param_spec != "d"):
                    print("Number of iterations (example: 300): ", end='')
                    niter = int(input())
                    print("Initial parameters (example: 1.57079633,1.57079633,1.57079633,1.57079633,1.57079633,1.57079633): ", end='')
                    init_params = input()
                    init_params = [float(item) for item in init_params.split(',')]
                    print("Stepsize(example 0.2): ", end='')
                    step_size = float(input())
                else: 
                    niter = 100
                    init_params = np.pi/2*np.ones(6)
                    step_size = 0.3  

                time_start = time.time()
                print("Running Adam---------------------------------------------------------------------------------------------")
                result = Adam(niter, init_params, step_size)
                time_stop = time.time()
                print("Adam plot saved to results directory")
                print("Adam time elapsed: " + str(time_stop - time_start))
                print()
                fig = plot_result(result, "Adam")



    
    
    
 