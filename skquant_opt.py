from skquant.opt import minimize
import matplotlib.pyplot as plt
import numpy as np

# some interesting objective function to minimize
# def objective_function_test(x):
#     fv = np.inner(x, x)
#     fv *= 1 + 0.1*np.sin(10*(x[0]+x[1]))
#     return np.random.normal(fv, 0.01)
# d = np.arange(0.0, 100.0, 0.01)
# T = np.reciprocal(d)
# plt.plot(d, T)  

def optimizer1(n, starting_point, total_test, Nparam, measure_list, pauli_list, define_VQE_ansatz, simulated_noise=False):
    
    def objective_function(x):
        E=-0.10312
        return E 

    weight=cal_weight(n,pauli_list)
    exact_E, Hfull=E_from_numpy(n,pauli_list)
    F=[]
    hp_a=[0.06,0.3]

    param=np.full(Nparam, 0.0)

    for T in range(starting_point, total_test):

        #the constants for SPSA
        a_n = hp_a[0] / np.power(T+1, hp_a[1])   
        c_n = 0.03 / np.power(T+1, 0.3)  # differential step size
        wol=1   # weight parameter for overlap in VQD
        #random the gradient estimation direction
        delta = np.random.binomial(1, 0.5, Nparam)*2-1
        
        list_dict=[]
        for _,mm in enumerate(measure_list):
            list_dict.append(get_VQE_result(n, param, simulated_noise,mm,define_VQE_ansatz))

        E_middle = evaluate(list_dict,weight)
        E_middle_exact = float(evaluate_with_Hamiltonian(n, param, Hfull,define_VQE_ansatz))
        F.append([E_middle,E_middle_exact])

        result, history = \
            minimize(objective_function, E_middle, bounds = np.array([[-1, 1], [-1, 1]], dtype=float) , budget=1, method='snobfit')  

        for i in range(Nparam):
            param[i] -= a_n * (result.optpar[1] - result.optpar[0]) / (2*c_n*delta[i])    
        
        print(result.optpar)     
        print(result.optval) 

    return F

def objective_function(x):
    E=-0.10312
    return E 

# create a numpy array of bounds, one (low, high) for each parameter
bounds = np.array([[-np.pi, np.pi], [-np.pi, np.pi]], dtype=float)   

# initial values for all parameters
#hp_a=[0.06,0.3] ?
x0 = np.array([0.0, 0.0])

# budget (number of calls, assuming 1 count per call)
budget = 1

# method can be ImFil, SnobFit, NOMAD, Orbit, or Bobyqa (case insensitive)
result, history = \
    minimize(objective_function, x0, bounds, budget, method='imfil')

# show results
print(result)  
# The result object will contain the optimal parameters (result.optpar) and optimal value (result.optval)
print(history)  
# The history object contains the full call history.
# plt.plot(history)
# plt.show()