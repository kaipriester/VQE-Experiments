#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:12:23 2021

@author: yangning8109
"""

import pennylane as qml
import numpy as np
from utils import getH, write_pauli_string

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
    return Efull,Hfull

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

def my_ansatz(p,w0,w1):
    qml.CNOT(wires=[w1,w0])
    qml.RY(p,wires=w1)
    qml.RZ(np.pi,wires=w1)
    qml.CNOT(wires=[w0,w1])
    qml.RZ(-np.pi,wires=w1)
    qml.RY(-p,wires=w1)
    qml.CNOT(wires=[w1,w0])
    return
    
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
    return
    

def make_circuit(param, measurenum=0, n=4):
    '''
    n: number of qubits:
    measurestr: measurement basis 0=ZZZZ 1=XXXX 2=YYYY
    '''
    # measurement circuit
    my_circuit(param)
    
    
    measurestr=''
    if measurenum==0: 
        measurestr='ZZZZ'
    elif measurenum==1: 
        measurestr='XXXX'
    elif measurenum==2: 
        measurestr='YYYY'
        
    #n=4
    for qb in range(n):
        if (measurestr[qb] == 'X'):  # Hadamard transform between X and Z
            qml.Hadamard(wires=qb)
        if (measurestr[qb] == 'Y'):  # transform
            qml.S(wires=qb)
            qml.Hadamard(wires=qb)
         
    return qml.probs(wires=[0,1,2,3])  

########################## main


def sim_run(params,n=4):
    #flag_vqe=1
    Udet=1.8
    tc=0.05
    n=4 # Qbit
    qml_jw,f_jw,qml_bk,f_bk, Hmatrix,Hbasis,H1body,H2body=getH(Ez1=1.,Ez2=0.9,Udet=Udet, tc=tc, U0=2.)
    write_pauli_string(qml_jw,filename="DQD_jw.txt",Nq=n)
    nbit=np.power(2,n)   # 2^n bit
    measure_list=['ZZZZ','XXXX','YYYY']
    ham_name='DQD_jw.txt' #the name of the file
    pauli_list=get_pauli_list(ham_name,measure_list)
    weight=cal_weight(n,pauli_list)  # weight for ZZZZ, XXXX, YYYY  from |0000> to |1111>

    exact_E, Hfull=E_from_numpy(n, pauli_list)  #exact_E[2]=target value


    dev = qml.device("default.qubit", wires = 4, shots = 80000)
    qnode=qml.QNode(make_circuit, dev)


    #params=np.pi/2*np.ones(6) # need update

    # 0 for ZZZZ ,1 for XXXX,2 for YYYY, ndarray = input parameters
    x0=qnode(params,0,n)
    x1=qnode(params,1,n)
    x2=qnode(params,2,n)

    sum_a=0
    for i,j in enumerate(x0):
        sum_a=sum_a+weight[0][i]*j
    for i,j in enumerate(x1):
        sum_a=sum_a+weight[1][i]*j
    for i,j in enumerate(x2):
        sum_a=sum_a+weight[2][i]*j
    
    print('F value:',sum_a)
    return sum_a
 
if __name__ == '__main__':
    params=np.pi/2*np.ones(6) # need update
    sim_run(params,n=4)

    
