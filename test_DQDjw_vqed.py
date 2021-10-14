#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:03:13 2021
Ref. https://github.com/TianyiPeng/Cluster-Simulation-Scheme/tree/master/VQE%20experiments

"""
# import qiskit and other useful python modules

#import math tools
import numpy as np
# We import plotting tools 
import matplotlib.pyplot as pl 
from   matplotlib.ticker import LinearLocator, FormatStrFormatter
# importing Qiskit
import qiskit as qk
import time
from utils import getH, write_pauli_string

from functions_common import E_from_numpy, run_SPSA

np.random.seed()

def get_pauli_list(ham_name,measure_list):
    input_file = open(ham_name) 
    pauli_list= [[] for _ in range(4)] # Z, X, Y,full
    ##read Hamiltonian from the file
    for lines in input_file:
        A = lines.split()[0]
        B = lines.split()[1][1:-1]
        pauli_list[-1].append([A, float(B)])
        
        #breakdown Pauli list
        if A.find('X')!=-1: #for 'XXXX' measurement
            flag=1
        elif A.find('Y')!=-1:   # 'YYYY' measurment
            flag=2
        else:
            flag=0  # ZZZZ
        pauli_list[flag].append([A, float(B)]) 

    return pauli_list

# generate the original, unreduced version of the quantum circuit for VQE

def A_ij():
    ### subcircuit definition, in https://arxiv.org/pdf/1904.10910.pdf
    # ptheta=theta+np.pi/2, pphi=phi+np.pi
    pphi=0+np.pi
    p=qk.circuit.Parameter('p')
    qca=qk.QuantumCircuit(2, name='A')
    qca.cx(1,0)
    qca.ry(p,1)
    qca.rz(pphi,1)
    qca.cx(0,1)
    qca.rz(-pphi,1)
    qca.ry(-p,1)
    qca.cx(1,0)
    return qca, p
    

def define_ansatz(n,param, list_1):
    qca, p=A_ij()
    
    ### quantum circuit for ansatz
    qr = qk.QuantumRegister(n, 'q')
    cr = qk.ClassicalRegister(n, 'c')
    qc = qk.QuantumCircuit(qr,cr)
    ### initialize the ansatz state
    for  ix in list_1:
        qc.x(qr[ix])
    param=np.array(param)+np.pi/2
    sub_ck=[qca.bind_parameters({p: p_val}) for p_val in param]
    sub_ck0=qca.bind_parameters({p: np.pi/2})
    qc.append(sub_ck[0].to_gate(),[qr[0],qr[1]])
    qc.append(sub_ck[1].to_gate(),[qr[2],qr[3]])
    qc.append(sub_ck0.to_gate(),[qr[1],qr[2]])
    qc.append(sub_ck[2].to_gate(),[qr[0],qr[1]])
    qc.append(sub_ck[3].to_gate(),[qr[2],qr[3]])
    qc.append(sub_ck0.to_gate(),[qr[1],qr[2]])
    qc.append(sub_ck[4].to_gate(),[qr[0],qr[1]])
    qc.append(sub_ck[5].to_gate(),[qr[2],qr[3]])
    return qc,qr, cr

def define_VQE_ansatz(n,param):
    list_1= [1,2]  # exited state is 1001 in Lu, Ru, Ld, Rd
    qc,qr,cr=define_ansatz(n,param, list_1)
    return qc,qr, cr

def ansatz(param, n):
    list_1= [1,2]  # exited state is 1001 in Lu, Ru, Ld, Rd
    qc,qr,cr=define_ansatz(n,param, list_1)
    return qc,qr, cr

def define_VQD_ansatz(n,param):
    list_1= [0,3]  # exited state is 1001 in Lu, Ru, Ld, Rd
    qc,qr,cr=define_ansatz(n,param, list_1)
    return qc,qr, cr

# def define_overlap_old(n,param): # https://arxiv.org/abs/1303.6814
#     ### quantum circuit for ansatz
#     nr=2*n+1
#     ### quantum circuit for ansatz
#     qr = qk.QuantumRegister(nr, 'q')
#     cr = qk.ClassicalRegister(1, 'c')
#     qc = qk.QuantumCircuit(qr,cr)

#     p_vqe=[-0.0025, -0.0094]  # need to manually update 
#     list_1=[1,2,4,7]
#     for ix in list_1:
#         qc.x(qr[ix])
    
#     qc.ry(p_vqe[0],qr[0])
#     qc.ry(p_vqe[1],qr[2])
    
#     qc.ry(param[0],qr[4])
#     qc.ry(param[1],qr[6])
    
#     control=2*n
#     qc.h(qr[control])
#     for i in range(n):
#         qc.cswap(qr[control],qr[i],qr[i+n])
#     qc.h(qr[control])
#     qc.barrier(qr)

#     qc.measure(qr[control],cr[0])   
#     return qc

def define_overlap(n,param): # https://arxiv.org/abs/1303.6814
    ### quantum circuit for ansatz
    nr=2*n+1
    qca, p=A_ij()
    ### quantum circuit for ansatz
    qr = qk.QuantumRegister(nr, 'q')
    cr = qk.ClassicalRegister(1, 'c')
    qc = qk.QuantumCircuit(qr,cr)

    p_in=[ 0.01353,  0.00313, -0.05454,  0.0013 ]  # need to manually update 
    list_1=[1,2,4,7]
    for ix in list_1:
        qc.x(qr[ix])

    p_vqe=np.array(p_in)+np.pi/2
    sub_ck=[qca.bind_parameters({p: p_val}) for p_val in p_vqe]
    qc.append(sub_ck[0].to_gate(),[qr[0],qr[1]])
    qc.append(sub_ck[1].to_gate(),[qr[2],qr[3]])
    sub_ck0=qca.bind_parameters({p: np.pi/2})
    qc.append(sub_ck0.to_gate(),[qr[1],qr[2]])
    qc.append(sub_ck[2].to_gate(),[qr[0],qr[1]])
    qc.append(sub_ck[3].to_gate(),[qr[2],qr[3]])
    
    p_vqd=np.array(param)+np.pi/2
    sub_ck=[qca.bind_parameters({p: p_val}) for p_val in p_vqd]
    qc.append(sub_ck[0].to_gate(),[qr[4],qr[5]])
    qc.append(sub_ck[1].to_gate(),[qr[6],qr[7]])
    sub_ck0=qca.bind_parameters({p: np.pi/2})
    qc.append(sub_ck0.to_gate(),[qr[5],qr[6]])
    qc.append(sub_ck[2].to_gate(),[qr[4],qr[5]])
    qc.append(sub_ck[3].to_gate(),[qr[6],qr[7]])
    
    control=2*n
    qc.h(qr[control])
    for i in range(n):
        qc.cswap(qr[control],qr[i],qr[i+n])
    qc.h(qr[control])
    qc.barrier(qr)

    qc.measure(qr[control],cr[0])   
    return qc

def sim_one(flag_vqe=1,Udet=1.8,tc=0.05):
    Nq=4   # number of qubits of Ansatz
    ### generate Pauli string according to Hamiltonian
    qml_jw,f_jw,qml_bk,f_bk, Hmatrix,Hbasis,H1body,H2body=getH(Ez1=1.,Ez2=0.9,
                                                        Udet=Udet, tc=tc, U0=2.)
    write_pauli_string(qml_jw,filename="DQD_jw.txt",Nq=Nq)
    
    ### set up meansurement and obtain Pauli string
    measure_list=['ZZZZ','XXXX','YYYY']
    ham_name='DQD_jw.txt' #the name of the file
    pauli_list=get_pauli_list(ham_name,measure_list)
    
    #CHANGING THIS-----------
    Nparam=6   # number of ansatz parameters
    ### print quantum circuits
    if flag_vqe==1:
        qc,qr,cr=define_VQE_ansatz(Nq,np.zeros(Nparam))
    else:
        qc,qr,cr=define_VQD_ansatz(Nq,np.zeros(Nparam))
    print(qc.draw())
    #qc=define_overlap(Nq,np.zeros(Nparam))
    #print(qc.draw())
    
    ### run single simlulation, for VQD add define_overlap
    if flag_vqe==1:
        F,Params,opt_param, opt_E,exact_E=run_SPSA(Nq,Nparam,measure_list,pauli_list,define_VQE_ansatz) 
        #run_opt(define_VQE_ansatz, Nparam, measure_list, pauli_list, define_overlap=None)
    else:  # VQD
        F,Params,opt_param, opt_E,exact_E=run_SPSA(Nq,Nparam,measure_list,pauli_list,define_VQD_ansatz,define_overlap)
    return F,Params,opt_param, opt_E,exact_E, Hmatrix, Hbasis
    
def sim_viz(flag_vqe=1, var_tc=1):
    Nq=4   # number of qubits of Ansatz
    
    Np=1   # number of simulation data points
    #NUMBER OF TIMES THE SIMULATION RUNS
    Ntry=1  # number of simulation tries
    Eexact, Esim=np.zeros((Np,Ntry)), np.zeros((Np,Ntry))
    Eexact_all=np.zeros((Np,Ntry, int(2**Nq)))
    Nparam=6   # number of ansatz parameters
    opt_par=np.zeros((Np,Ntry,Nparam))

    if var_tc==1:  # iterate over tc
        xx=np.linspace(0.01,0.08,Np)
        xlabel='$t_c$'
        for itry in range(Ntry):
            for ii, par in enumerate(xx):
                F,Params,opt_param, opt_E,exact_E,_,_=sim_one(flag_vqe=flag_vqe,tc=par)
                idx = (np.abs(exact_E - opt_E[0])).argmin()
                Eexact[ii,itry]=exact_E[idx]
                Eexact_all[ii,itry]=exact_E
                Esim[ii,itry]=opt_E[0]
                opt_par[ii,itry]=opt_param
    else:  # vary Udet
        xx=np.linspace(1.0,1.8,Np)
        xlabel='$U_d$'
        for itry in range(Ntry):
            for ii, par in enumerate(xx):
                F,Params,opt_param, opt_E,exact_E,_,_=sim_one(flag_vqe=flag_vqe,Udet=par)
                idx = (np.abs(exact_E - opt_E[0])).argmin()
                Eexact[ii,itry]=exact_E[idx]
                Eexact_all[ii,itry]=exact_E
                Esim[ii,itry]=opt_E[0]
                opt_par[ii,itry]=opt_param
                print('U=',par)
    ### visualize
    pl.figure()
    pl.plot(xx,np.min(Esim,axis=1),'b-',lw=3)
    pl.plot(xx,np.min(Eexact,axis=1),'ro',markersize=12)
    pl.xlabel(xlabel)
    pl.ylabel('E')
    return xx,Esim,Eexact,Eexact_all, opt_par

def save_result(fn,xx,Esim,Eexact,opt_par):
    import os.path
    if os.path.exists(fn):  # combine with existing data
        data = np.load(fn)
        if np.all(data['xx']==xx):
            Esim=np.hstack((Esim, data['Esim']))
            Eexact=np.hstack((Eexact, data['Eexact']))
            opt_par=np.concatenate((opt_par, data['opt_par']), axis = 1)
    np.savez(fn, xx=xx, Esim=Esim, Eexact=Eexact, opt_par=opt_par)
    return xx,Esim,Eexact,opt_par
    

# By default, we run this circuit on the IBM 16-qubit machine or corresponding simulator
if __name__ == '__main__':
    np.set_printoptions(precision=5)
    ### individual simulation
    #F,Params,opt_param, opt_E,exact_E, Hmatrix,Hbasis = sim_one(flag_vqe=1)
    ### batch simulation
    xx,Esim,Eexact,Eexact_all,opt_par=sim_viz(flag_vqe=1,var_tc=0)
    #xx,Esim,Eexact,Eexact_all,opt_par=save_result('results/vqe_DQD_U.npz',xx,Esim,Eexact,opt_par)
    