#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:03:13 2021
Basis set is up 1,2,3,4 , dn1,2,3,4

"""
# import qiskit and other useful python modules

#import math tools
import numpy as np
# import plotting tools 
import matplotlib.pyplot as plt 
from   matplotlib.ticker import LinearLocator, FormatStrFormatter
# importing Qiskit
import qiskit as qk
import time

from functions_common import E_from_numpy, run_SPSA 
from utils import getH4,  write_pauli_string
import matplotlib.pyplot as pl 

np.random.seed(int(time.time()))

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

def A_ij():
    ### subcircuit definition, use ansatz in last figure of https://arxiv.org/pdf/1904.10910.pdf
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

def define_VQE_ansatz(n,param):
    qca, p=A_ij()
    
    ### quantum circuit for ansatz
    qr = qk.QuantumRegister(n, 'q')
    cr = qk.ClassicalRegister(n, 'c')
    qc = qk.QuantumCircuit(qr,cr)
    ### initialize the ansatz state
    list_1=[1,3,4,6]  # exited state is 1001 in Lu, Ru, Ld, Rd
    for  ix in list_1:
        qc.x(qr[ix])
    param=np.array(param)+np.pi/2
    sub_ck=[qca.bind_parameters({p: p_val}) for p_val in param]
    sub_ck0=qca.bind_parameters({p: np.pi/2})
    qc.append(sub_ck0.to_gate(),[qr[3],qr[4]])
    qc.append(sub_ck[0].to_gate(),[qr[2],qr[3]])
    qc.append(sub_ck[1].to_gate(),[qr[4],qr[5]])
    qc.append(sub_ck[2].to_gate(),[qr[1],qr[2]])
    qc.append(sub_ck[3].to_gate(),[qr[5],qr[6]])
    qc.append(sub_ck0.to_gate(),[qr[3],qr[4]])
    
    qc.append(sub_ck[4].to_gate(),[qr[0],qr[1]])
    qc.append(sub_ck[5].to_gate(),[qr[2],qr[3]])
    qc.append(sub_ck[6].to_gate(),[qr[4],qr[5]])
    qc.append(sub_ck[7].to_gate(),[qr[6],qr[7]])
    
    qc.append(sub_ck[8].to_gate(),[qr[1],qr[2]])
    qc.append(sub_ck[9].to_gate(),[qr[5],qr[6]])
    qc.append(sub_ck0.to_gate(),[qr[3],qr[4]])
    
    qc.append(sub_ck[10].to_gate(),[qr[2],qr[3]])
    qc.append(sub_ck[11].to_gate(),[qr[4],qr[5]])
    qc.append(sub_ck0.to_gate(),[qr[3],qr[4]])
    return qc,qr, cr

def define_VQE_ansatz1(n,param):
    qca, p=A_ij()
    
    ### quantum circuit for ansatz
    qr = qk.QuantumRegister(n, 'q')
    cr = qk.ClassicalRegister(n, 'c')
    qc = qk.QuantumCircuit(qr,cr)
    ### initialize the ansatz state
    list_1=[1,3,4,6]  # exited state is 1001 in Lu, Ru, Ld, Rd
    for  ix in list_1:
        qc.x(qr[ix])
    param=np.array(param)+np.pi/2
    sub_ck=[qca.bind_parameters({p: p_val}) for p_val in param]
    sub_ck0=qca.bind_parameters({p: np.pi/2})
    qc.append(sub_ck[0].to_gate(),[qr[0],qr[1]])
    qc.append(sub_ck[1].to_gate(),[qr[1],qr[2]])
    qc.append(sub_ck[2].to_gate(),[qr[2],qr[3]])
    qc.append(sub_ck[3].to_gate(),[qr[7],qr[6]])
    qc.append(sub_ck[4].to_gate(),[qr[6],qr[5]])
    qc.append(sub_ck[5].to_gate(),[qr[5],qr[4]])
    qc.append(sub_ck0.to_gate(),[qr[3],qr[4]])
    qc.append(sub_ck[6].to_gate(),[qr[2],qr[3]])
    qc.append(sub_ck[7].to_gate(),[qr[1],qr[2]])
    qc.append(sub_ck[8].to_gate(),[qr[0],qr[1]])
    qc.append(sub_ck[9].to_gate(),[qr[5],qr[4]])
    qc.append(sub_ck[10].to_gate(),[qr[6],qr[5]])
    qc.append(sub_ck[11].to_gate(),[qr[7],qr[6]])
    return qc,qr, cr

def sim_one(Udet=1.8,tc=0.05):
    Nq=8   # number of qubits of Ansatz
    ### generate Pauli string according to Hamiltonian
    qml_jw,f_jw,qml_bk,f_bk, Hmatrix,Hbasis,H1body,H2body=getH4(Ez=[1.,0.9,1.,0.9]
                                        ,Udet=1.8, tc=tc,U0=2.)
    write_pauli_string(qml_jw,filename="Q4_jw.txt",Nq=Nq)
    
    measure_list=['ZZZZZZZZ','XXXXXXXX','YYYYYYYY']   # JW Pauli meassurments
    ham_name='Q4_jw.txt' #the name of the file
    pauli_list=get_pauli_list(ham_name,measure_list)

    Nparam=12   # number of ansatz parameters
    ### print quantum circuits
    qc,qr,cr=define_VQE_ansatz(Nq,np.zeros(Nparam))
    print(qc.draw())
    
    ### run single simlulation, for VQD add define_overlap
    F,Params,opt_param, opt_E,exact_E=run_SPSA(Nq,Nparam,measure_list,pauli_list,define_VQE_ansatz) 
  
    return F,Params,opt_param, opt_E,exact_E,Hmatrix,Hbasis

def sim_viz( var_tc=1):
    Nq=8   # number of qubits of Ansatz
    
    Np=5   # number of simulation data points
    Ntry=1  # number of simulation tries
    Eexact, Esim=np.zeros(Np), np.zeros((Np,Ntry))
    Eexact_all=np.zeros((Np,int(2**Nq)))
    Nparam=12   # number of ansatz parameters
    opt_par=np.zeros((Np,Ntry,Nparam))

    if var_tc==1:  # iterate over tc
        xx=np.linspace(0.01,0.05,Np)
        xlabel='$t_c$'
        for itry in range(Ntry):
            for ii, par in enumerate(xx):
                F,Params,opt_param, opt_E,exact_E,_,_=sim_one(tc=par)
                idx = (np.abs(exact_E - opt_E[0])).argmin()
                Eexact[ii]=exact_E[idx]
                Eexact_all[ii]=exact_E
                Esim[ii,itry]=opt_E[0]
                opt_par[ii,itry]=opt_param
    else:  # vary Udet
        xx=np.linspace(1.5,1.9,Np)
        xlabel='$U_d$'
        for itry in range(Ntry):
            for ii, par in enumerate(xx):
                F,Params,opt_param, opt_E,exact_E,_,_=sim_one(Udet=par)
                idx = (np.abs(exact_E - opt_E[0])).argmin()
                Eexact[ii]=exact_E[idx]
                Eexact_all[ii]=exact_E
                Esim[ii,itry]=opt_E[0]
                opt_par[ii,itry]=opt_param
                print('U=',par)
    ### visualize
    pl.figure()
    pl.plot(xx,np.min(Esim,axis=1),'b-',lw=3)
    pl.plot(xx,Eexact,'ro',markersize=12)
    pl.xlabel(xlabel)
    pl.ylabel('E')
    
    return xx,Esim,Eexact,Eexact_all, opt_par

if __name__ == '__main__':
    np.set_printoptions(precision=4)
    ### individual simulation
    #F,Params,opt_param, opt_E,exact_E,Hmatrix,Hbasis = sim_one()
    
    ### sbatch simulation
    xx,Esim,Eexact,Eexact_all,opt_par=sim_viz(var_tc=1)
   
