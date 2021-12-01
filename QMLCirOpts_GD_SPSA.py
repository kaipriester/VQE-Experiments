""" Kai Priester
    Fall 2021
    //working CLI to run different opts on circuit defined in pennyl3 """
    
import pennylane as qml
from noisyopt import minimizeSPSA
import numpy as np
import matplotlib.pyplot as plt 
import time

from pennyl3 import sim_run

def GD(steps, init_params, step_size, type):
    if(type == "GD"): 
        opt = qml.GradientDescentOptimizer(stepsize=step_size)   
    elif(type == "Adam"):
        opt = qml.AdamOptimizer(stepsize=step_size)  
    elif(type == "Ada"):
        opt = qml.AdagradOptimizer(stepsize=step_size)  
    elif(type == "MO"):
        opt = qml.MomentumOptimizer(stepsize=step_size)  
    elif(type == "NMO"):
        opt = qml.NesterovMomentumOptimizer(stepsize=step_size)      
    elif(type == "RMSProp"):
        opt = qml.RMSPropOptimizer(stepsize=step_size)   

    cost = sim_run

    params = init_params

    F = []
    for k in range(steps):
        print("Step: " + str(k))
        params, energy = opt.step_and_cost(cost, params)
        F.append(energy)

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

#new -0.10911
def plot_result(F, opt_name, exact_E=-0.10911):
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
        print("     4. Gradient-descent optimizer with past-gradient-dependent learning rate in each dimension. (Ada)")
        print("     5. Gradient-descent optimizer with momentum. (MO)")
        print("     6. Gradient-descent optimizer with Nesterov momentum. (NMO)")
        print("     7. Root mean squared propagation optimizer. (RMSProp)")
        print("Input menu ID(s) of optimizers to run (examples: 1,3 or 2): ", end='')
        menu_input = input()   
        menu_input = list(menu_input.split(","))

        for i in menu_input:
            if(i == '1'):
                valid_input = True
                print("GD hyperparameters (input d to use defaults OR any other key to continue modifying): ", end='')
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
                result = GD(niter, init_params, step_size, "GD")
                time_stop = time.time()
                print("GD plot saved to results directory")
                print("GD time elapsed: " + str(time_stop - time_start))
                print()
                plot_result(result, "GD")

            elif(i == '2'):
                valid_input = True
                print("SPSA hyperparameters (input d to use defaults OR any other key to continue modifying): ", end='')
                param_spec = input()
                if(param_spec != "d"):
                    print("Number of iterations (example: 300): ", end='')
                    niter = int(input())
                    print("Initial parameters (example: 1.57079633,1.57079633,1.57079633,1.57079633,1.57079633,1.57079633): ", end='')
                    init_params = input()
                    init_params = [float(item) for item in init_params.split(',')]
                    print("c (example 0.2): ", end='')
                    c = float(input())
                    print("a (example 1.0): ", end='')
                    a = float(input())

                else: 
                    niter = 100
                    init_params = np.pi/2*np.ones(6)
                    c = 0.2
                    a = 1.0  

                time_start = time.time()
                print("Running SPSA---------------------------------------------------------------------------------------------")
                result = SPSA(niter, init_params, c, a)
                time_stop = time.time()
                print("SPSA plot saved to results directory")
                print("SPSA time elapsed: " + str(time_stop - time_start))
                print()
                plot_result(result, "SPSA")

            elif(i == '3'):
                valid_input = True
                print("Adam hyperparameters (input d to use defaults OR any other key to continue modifying): ", end='')
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
                result = GD(niter, init_params, step_size, "Adam")
                time_stop = time.time()
                print("Adam plot saved to results directory")
                print("Adam time elapsed: " + str(time_stop - time_start))
                print()
                plot_result(result, "Adam")

            elif(i == '4'):
                valid_input = True
                print("Ada hyperparameters (input d to use defaults OR any other key to continue modifying): ", end='')
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
                print("Running Ada---------------------------------------------------------------------------------------------")
                result = GD(niter, init_params, step_size, "Ada")
                time_stop = time.time()
                print("Ada plot saved to results directory")
                print("Ada time elapsed: " + str(time_stop - time_start))
                print()
                plot_result(result, "Ada")    

            elif(i == '5'):
                valid_input = True
                print("MO hyperparameters (input d to use defaults OR any other key to continue modifying): ", end='')
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
                print("Running MO---------------------------------------------------------------------------------------------")
                result = GD(niter, init_params, step_size, "MO")
                time_stop = time.time()
                print("MO plot saved to results directory")
                print("MO time elapsed: " + str(time_stop - time_start))
                print()
                plot_result(result, "MO")     

            elif(i == '6'):
                valid_input = True
                print("NMO hyperparameters (input d to use defaults OR any other key to continue modifying): ", end='')
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
                print("Running NMO---------------------------------------------------------------------------------------------")
                result = GD(niter, init_params, step_size, "NMO")
                time_stop = time.time()
                print("NMO plot saved to results directory")
                print("NMO time elapsed: " + str(time_stop - time_start))
                print()
                plot_result(result, "NMO")   

            
            elif(i == '7'):
                valid_input = True
                print("RMSProp hyperparameters (input d to use defaults OR any other key to continue modifying): ", end='')
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
                print("Running RMSProp---------------------------------------------------------------------------------------------")
                result = GD(niter, init_params, step_size, "RMSProp")
                time_stop = time.time()
                print("RMSProp plot saved to results directory")
                print("RMSProp time elapsed: " + str(time_stop - time_start))
                print()
                plot_result(result, "RMSProp")    
                            

    
    
    
 