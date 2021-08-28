#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:03:13 2021
Ref. https://github.com/TianyiPeng/Cluster-Simulation-Scheme/tree/master/VQE%20experiments

"""
# import qiskit and other useful python modules

#plot is directly shown inline

#import math tools
import numpy as np
# We import plotting tools 
import matplotlib.pyplot as plt 
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter
# importing Qiskit
import qiskit as qk
import time
from qiskit.visualization import plot_histogram
from qiskit.providers.aer import noise

import pickle
import copy

### print all backends: print(provider.backends())

# obtain sampling distribution from the original, unreduced version of the quantum circuit for VQE
# By default, we run this circuit on the IBM 16-qubit machine or corresponding simulator
## 16-qubit backend
qk.IBMQ.load_account()
provider = qk.IBMQ.get_provider(group='open')

#device_16 = provider.get_backend('ibmq_16_melbourne')
device_Q=qk.Aer.get_backend('qasm_simulator')
properties = device_Q.properties()

# noise model 
noise_model_Q = noise.NoiseModel()
#noise_model_Q = noise.NoiseModel.from_backend(device_Q) # for real device

# Get coupling map from backend
coupling_map_Q = device_Q.configuration().coupling_map
# Get basis gates from noise model
basis_gates_Q = noise_model_Q.basis_gates


### initialize NSHOTS 
NSHOTS=80000


def E_from_numpy(Nq,pauli_list):
    I = np.matrix([[1, 0], [0, 1]])
    X = np.matrix([[0, 1], [1, 0]])
    Y = np.matrix([[0, -1j], [1j, 0]])
    Z = np.matrix([[1, 0], [0, -1]])
    
    pauli_list_full=pauli_list[-1]
    pauli_list_trun=pauli_list[0]

    #compute the energy of initialized Hamiltonian
    Hfull = np.zeros([2**Nq, 2**Nq])
    for i in range(len(pauli_list_full)):
        basis = pauli_list_full[i][0] 
        coef = pauli_list_full[i][1]
        A = np.matrix(1)
        for s in basis:
            if (s == 'I'):
                A = np.kron(A, I)
            else:
                if (s == 'X'):
                    A = np.kron(A, X)
                else:
                    if (s == 'Z'):
                        A = np.kron(A, Z)
                    else:
                        if (s == 'Y'):
                            A = np.kron(A, Y)
        Hfull = Hfull + A*coef
    Efull, V = np.linalg.eig(Hfull)
    Efull=Efull.real
    Efull=Efull[np.argsort(Efull)]
    print('E of full Pauli string:',Efull)

    
    ##compute the energy of the truncated Hamiltonian
    # Htrun = np.zeros([2**Nq, 2**Nq])
    # for i in range(len(pauli_list_trun)):
    #     basis = pauli_list_trun[i][0] 
    #     coef = pauli_list_trun[i][1]
    #     A = np.matrix(1)
    #     for s in basis:
    #         if (s == 'I'):
    #             A = np.kron(A, I)
    #         else:
    #             if (s == 'X'):
    #                 A = np.kron(A, X)
    #             else:
    #                 if (s == 'Z'):
    #                     A = np.kron(A, Z)
    #                 else:
    #                     if (s == 'Y'):
    #                         A = np.kron(A, Y)
    #     Htrun = Htrun + A*coef
    # Etrun, V = np.linalg.eig(Htrun)
    # idx = np.argsort(Etrun)
    # print('truncated E:',Etrun[idx])
    return Efull,Hfull

def evaluate_with_Hamiltonian(n, param, H, define_VQE_ansatz):
    '''
    Given the VQE quantum circuit and Hamiltonian, return the associated 'ideal' value of observable 
    
    n: number of qubits
    depth: depth of the circuit (currently we only allow depth = 1)
    param: parameters describing the circuit
    H: Hamiltonian
    '''
    
    ### quantum circuit for ansatz
    qc,qr,cr=define_VQE_ansatz(n,param)
    
    # Select the StatevectorSimulator from the Aer provider
    simulator = qk.Aer.get_backend('statevector_simulator')

    # Execute and get the resulting state vector
    result = qk.execute(qc, simulator).result()
    vector = result.get_statevector(qc)
    statevector = np.zeros(2**n, dtype='complex')
    for idx in range(2**n):
        bit_string = format(idx, "0%db" % n)
        statevector[int(bit_string[::-1],2)] = vector[idx]
    statevector = np.matrix(statevector)
    # Return the observable 
    return np.trace(statevector*H*statevector.getH())

##Compute the evaluation function, only apply to Z measurement
# input, pauli_list, output, map binary state to its energy
def cal_one_weight(Nq, pauli_list): # sume over Pauli list
    weight = np.zeros([2**Nq])
    for idx in range(2**Nq):
        bit_string = format(idx, '0%db' % Nq)
        for (basis, coef) in pauli_list:  # iterate over pauli_list
            v = 1
            for i in range(len(bit_string)):
                if (basis[i] != 'I'):
                    v *= 1-2*int(bit_string[i])
            weight[idx] += v*coef
    return weight   # 2**Nq array

def cal_weight(Nq,measure_list_of_pauli_list):  # wrapper for multiple measurment 
    ml_pl=measure_list_of_pauli_list
    Nw=len(ml_pl)-1
    list_of_weight=[[] for _ in range(Nw)] # list (length of measr.) of weight arrays (array in Fock space) 
    for ii in range(Nw):
        list_of_weight[ii]=cal_one_weight(Nq,ml_pl[ii])
    return list_of_weight

 
def evaluate(list_of_ddict, list_of_weight):  # sumed over measurement distribution
    sum_pauli_terms=0
    for ii, ddict in enumerate(list_of_ddict): # sum over list of measurement dictionaries 
        ntot=sum(ddict.values())
        if len(ddict)>0:  # sum over the ith measurment distribution
            for bit_string in ddict:
                sum_pauli_terms += list_of_weight[ii][int(bit_string, 2)]*ddict[bit_string] /ntot
    return sum_pauli_terms

def make_VQE_circuit(n, param, measurement, define_VQE_ansatz):
    '''
    n: number of qubits:
    depth: depth of circuits; 
    thetas: the control parameters for VQE; dimension = (n,depth*3+2)
    measurement: measurement basis
    '''
    # construct ansatz
    qc,qr,cr=define_VQE_ansatz(n,param)
    
    # measurement circuit
    qc.barrier(qr)
    for qb in range(n):
        if (measurement[qb] == 'X'):  # Hadamard transform between X and Z
            qc.h(qr[qb])
        if (measurement[qb] == 'Y'):  # transform
            qc.sdg(qr[qb])
            qc.h(qr[qb])
        qc.measure(qr[qb],cr[qb])    
    return qc,qr

def get_VQE_result(n, default_thetas, simulated_noise=False, measurement='ZZZZ',
                   define_VQE_ansatz=[],nshots=NSHOTS):
    '''
    n: number of qubits:
    depth: number of entanglements; 
    default_thetas: the control parameters for VQE; dimension = (n,depth)
    backend: backend the of quantum circuit (simulator or real device)
    simulated_noise: whether adding noise for the simulator
    measurement: measurement basis
    '''
    qc,qr = make_VQE_circuit(n, default_thetas, measurement,define_VQE_ansatz)
    if (simulated_noise):
        job = qk.execute(qc, device_Q,shots=nshots, coupling_map=coupling_map_Q, noise_model=noise_model_Q, 
                      basis_gates=basis_gates_Q) 
                    #,initial_layout={qr[3] : 2, qr[2] : 1, qr[1] : 0, qr[0] : 14})
    else:
        job = qk.execute(qc, device_Q, shots=nshots) #,initial_layout={qr[3] : 2, qr[2] : 1, qr[1] : 0, qr[0] : 14})

    result = job.result()
    ddict_tmp = result.get_counts(0)
    
    #reverse the output, index from 0-th qubit 
    ddict = {}
    for bit_string in ddict_tmp:
        ddict[bit_string[::-1]] = ddict_tmp[bit_string]
    return ddict

def get_overlap(n, default_thetas,  simulated_noise=False,overlap_ck=[]):
    ### See https://arxiv.org/pdf/1303.6814.pdf
    qc=overlap_ck(n,default_thetas)
    if (simulated_noise):
        job = qk.execute(qc, device_Q,shots=NSHOTS, coupling_map=coupling_map_Q, noise_model=noise_model_Q, 
                      basis_gates=basis_gates_Q) 
                    #,initial_layout={qr[3] : 2, qr[2] : 1, qr[1] : 0, qr[0] : 14})
    else:
        job = qk.execute(qc, device_Q, shots=NSHOTS) #,initial_layout={qr[3] : 2, qr[2] : 1, qr[1] : 0, qr[0] : 14})

    result = job.result()
    ddict = result.get_counts(0)
    overlap=ddict['0']/NSHOTS-ddict['1']/NSHOTS # P(0)-P(1) is overlap |<a|b>|^2
    overlap=np.maximum(overlap,0)  # postprocess to remove noise and error
    return overlap

def SPSA_vqd_or_vqe(n, starting_point, total_test, define_VQE_ansatz, simulated_noise=False, 
                      file_name = "params.txt", 
                      param=[0.,0.],pauli_list=[],measure_list=[],define_overlap=None):
    '''
    n: number of qubits
    depth: depth of the circuit
    starting_point: for debuging, by default, 0
    total_test: the number of iterations
    get_VQE_result: a function reference for choosing the appropriate algorithms
    simulated_noise: adding noise to the simulator
    file_name: the running data is stored in the file
    device: the simulator or the real quantum chip
    '''
    
    weight=cal_weight(n,pauli_list)
    exact_E, Hfull=E_from_numpy(n,pauli_list)
    
    # the following global definitions are in place because
    # we still want to access the last running result
    # in case any error occurs during optimizatrion
    F=[]
    Params=[]

    depth = 1
    
    L = len(param) # the total number of parameters
    beta = np.reshape(param,(-1,1)) #initilize the parameters
    
    # if (starting_point > 0): #for debugging, start for a specified parameters path
    #     total = 0
    #     for i in range(n):
    #         for j in range(depth*3+2):
    #             beta[total] = Params[starting_point][total]
    #             total = total + 1

    hp_a=[0.06,0.3]   # SPSA hyperparameter,  hp_a=[0.06,0.3] for Q4 and DQD
    for T in range(starting_point, total_test):
        
        #the constants for SPSA
        a_n = hp_a[0] / np.power(T+1, hp_a[1])   
        c_n = 0.03 / np.power(T+1, 0.3)  # differential step size
        wol=1   # weight parameter for overlap in VQD
        
        #random the gradient estimation direction
        delta = np.random.binomial(1, 0.5, L)*2-1
                
        #evaluate the plus direction
        #for F_plus
        for i in range(L):
            param[i]=(beta[i]+delta[i]*c_n).tolist()[0]
        list_dict=[]
        for _,mm in enumerate(measure_list):
            list_dict.append(get_VQE_result(n, param, simulated_noise,mm,define_VQE_ansatz))
        if define_overlap is None:
            F_plus = evaluate(list_dict,weight)
        else:   # VQD
            ol=get_overlap(n, param, simulated_noise,define_overlap)
            F_plus=wol*ol+evaluate(list_dict,weight)
        
        #for F_real_plus
        #value_plus = float(evaluate_with_Hamiltonian(n, param, Hfull,define_VQE_ansatz))

        #evaluate the minus direction
        #for F_minus
        for i in range(L):
            param[i]=(beta[i]-delta[i]*c_n).tolist()[0]
        list_dict=[]
        for _,mm in enumerate(measure_list):
            list_dict.append(get_VQE_result(n, param,  simulated_noise,mm,define_VQE_ansatz))
        if define_overlap is None:
            F_minus = evaluate(list_dict,weight)
        else:   # VQD
            ol=get_overlap(n, param, simulated_noise,define_overlap)
            F_minus=wol*ol+evaluate(list_dict,weight)
        
        #for F_real_minus
        #value_minus = float(evaluate_with_Hamiltonian(n, param, Hfull,define_VQE_ansatz))
        
        ### calculate F value
        param=beta.ravel()
        list_dict=[]
        for _,mm in enumerate(measure_list):
            list_dict.append(get_VQE_result(n, param, simulated_noise,mm,define_VQE_ansatz))
        E_middle = evaluate(list_dict,weight)
        
        E_middle_exact = float(evaluate_with_Hamiltonian(n, param, Hfull,define_VQE_ansatz))
 
        #store the parameters in the current step
        Params.append(copy.deepcopy(beta.ravel()))   
        F.append([E_middle,E_middle_exact])
        fw = open(file_name, 'wb')
        pickle.dump([Params, F], fw)

        print('At step {0}, the F_minus is {1:.5f}, the F_plus is {2:.5f}, and the gradient scale \
              is {3:.5f}\n'.format(T, F_minus, F_plus, a_n*(F_plus-F_minus)/c_n))
        # print('At step {0}, the F_real_minus is {1:.5f}, the F_real_plus is {2:.5f}, and the \
        # real gradient scale is {3:.5f}\n'.format(T, value_minus, value_plus, a_n*(value_plus-value_minus)/c_n))
        #update the parameters
        for i in range(L):
            beta[i] -= a_n * (F_plus - F_minus) / (2*c_n*delta[i])
    
    ### get the optimum parameter
    F=np.array(F)
    Params=np.array(Params)
    iopt=F[:,1].argmin()  # index of the identified optimium
    opt_param=Params[iopt]
    opt_E=F[iopt,0:2] # energy, QC value, numpy value
    # refined calculation of opt_E by using a larger NSHOTS
    param=opt_param
    list_dict=[]
    for _,mm in enumerate(measure_list):
        list_dict.append(get_VQE_result(n, param, simulated_noise,mm,define_VQE_ansatz,nshots=400000))
    opt_E[0] = evaluate(list_dict,weight)
    if define_overlap is not None:  # print the overlap for checking VQD solution
        print('overlap is', ol)
    return np.array(F), np.array(Params),opt_param, opt_E, exact_E

## wrapper for running simulation
def run_SPSA(n,Nparam,measure_list,pauli_list,define_VQE_ansatz, define_overlap=None):
    ### run the experiment, time consuming if run on ibm-q cloud (several days)
    # measure: 'rho' or 'pauli'
    Params = []
    F = []
    #NUMBER OF STEPS
    Niter=10      # total iteration number
    F, Params,opt_param, opt_E, exact_E = SPSA_vqd_or_vqe(n, 0, Niter, define_VQE_ansatz, simulated_noise=True, 
            file_name='SPSA_result_orig.txt', 
            pauli_list=pauli_list,measure_list=measure_list,param=np.zeros(Nparam),
            define_overlap=define_overlap)
    
    ### plot results
    #exact_E=-0.1026  # need to edit for every case
    #should converge to exact_E
    def plot_result(F):
        Nitr=len(F)
        plt.figure()
        xv=np.arange(Nitr)+1
        plt.plot(xv,np.array(F)[:,0])
        #plt.plot(xv,exact_E*np.ones(Nitr),'g--')
        plt.xlabel('iteration number')
        plt.ylabel('energy')
        plt.show()
    plot_result(F)
    return F,Params, opt_param, opt_E, exact_E


### old SPSA optimizer (not used, just for reference)
# # the simultaneous perturbation stochastic approximation (SPSA) method
# # described in https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation 
# def SPSA_optimization(n, starting_point, total_test, define_VQE_ansatz, simulated_noise=False, 
#                       file_name = "params.txt", device=qk.Aer.get_backend('qasm_simulator'),
#                       param=[0.,0.],pauli_list=[],measure_list=[]):
#     '''
#     n: number of qubits
#     depth: depth of the circuit
#     starting_point: for debuging, by default, 0
#     total_test: the number of iterations
#     get_VQE_result: a function reference for choosing the appropriate algorithms
#     simulated_noise: adding noise to the simulator
#     file_name: the running data is stored in the file
#     device: the simulator or the real quantum chip
#     '''
    
#     weight=cal_weight(n,pauli_list)
#     Hfull, Htrun, Efull, Etrun=E_from_numpy(n,pauli_list)
    
#     # the following global definitions are in place because
#     # we still want to access the last running result
#     # in case any error occurs during optimizatrion
#     F=[]
#     Params=[]

#     depth = 1
    
#     L = 2 # the total number of parameters
#     beta = np.reshape(param,(-1,1)) #initilize the parameters
    
#     if (starting_point > 0): #for debugging, start for a specified parameters path
#         total = 0
#         for i in range(n):
#             for j in range(depth*3+2):
#                 beta[total] = Params[starting_point][total]
#                 total = total + 1

    
#     for T in range(starting_point, total_test):
        
#         #the constants for SPSA
#         an = 0.2 / np.power(T+1, 0.3)
#         cn = 0.06 / np.power(T+1, 0.5)
        
#         #random the gradient estimation direction
#         delta = np.random.binomial(1, 0.5, L)*2-1
                
#         #evaluate the plus direction
#         for i in range(L):
#             param[i]=(beta[i]+delta[i]*cn).tolist()[0]
#         ddict=[]
#         for _,mm in enumerate(measure_list):
#             ddict.append(get_VQE_result(n, param, device, simulated_noise,mm,define_VQE_ansatz))
#         F_plus = evaluate(ddict,weight)
        
#         value_plus = float(evaluate_with_Hamiltonian(n, param, Hfull,define_VQE_ansatz))

#         #evaluate the minus direction
#         for i in range(L):
#             param[i]=(beta[i]-delta[i]*cn).tolist()[0]
#         ddict=[]
#         for _,mm in enumerate(measure_list):
#             ddict.append(get_VQE_result(n, param, device, simulated_noise,mm,define_VQE_ansatz))
#         F_minus = evaluate(ddict,weight)

#         value_minus = float(evaluate_with_Hamiltonian(n, param, Hfull,define_VQE_ansatz))
        
#         ### calculate F value
#         param=beta.ravel()
#         ddict=[]
#         for _,mm in enumerate(measure_list):
#             ddict.append(get_VQE_result(n, param, device, simulated_noise,mm,define_VQE_ansatz))
#         F_middle = evaluate(ddict,weight)
#         value_middle = float(evaluate_with_Hamiltonian(n, param, Hfull,define_VQE_ansatz))

#         #update the parameters
#         for i in range(L):
#             beta[i] -= an * (F_plus - F_minus) / (2*cn*delta[i])
            
#         #store the parameters in the current step
#         Params.append(copy.deepcopy(beta))   
#         F.append([F_middle,value_middle,F_minus, value_minus, F_plus, value_plus])
#         fw = open(file_name, 'wb')
#         pickle.dump([Params, F], fw)

        
#         print('At step {0}, the F_minus is {1:.5f}, the F_plus is {2:.5f}, and the gradient scale \
#               is {3:.5f}\n'.format(T, F_minus, F_plus, an*(F_plus-F_minus)/cn))
#         print('At step {0}, the F_real_minus is {1:.5f}, the F_real_plus is {2:.5f}, and the \
#      real gradient scale is {3:.5f}\n'.format(T, value_minus, value_plus, an*(value_plus-value_minus)/cn))
#         opt_param=beta
#     return F, Params,opt_param

