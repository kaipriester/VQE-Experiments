#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:01:14 2021

@author: jingguo
"""

from openfermion import *
from pennylane import qchem  # for conversion from optenfermion to pennylane
import numpy as np
### convert the 2nd quantization Hamiltonian to Pauli string Hamiltonian

def getH(Ez1=1.,Ez2=0.9,Udet=9.1, tc=0.05, U0=10.):  # mapping to Pauli string Hamiltonian
    N=4
    #UL=U0-Udet
    #UR=U0+Udet
    kd=np.kron(np.array([1.,-1.]),np.array([Ez1,Ez2]))   # Zeeman
    kd+=Udet/2*np.kron(np.array([1.,1.]),np.array([-1., 1.]))   # detuning
    kappa=np.zeros((N,N))
    kappa[0,1]=1.   # between up spin |Lu> and |Ru>
    kappa[1,0]=1.
    kappa[2,3]=-1.  # between down spin |Lu> and |Ru>
    kappa[3,2]=-1.
    kappa=tc*kappa   # tunnel coupling
    kappa+=np.diag(kd) # 1-body term
    
    ### 2 body interaction terms
    k2=np.zeros((N, N, N, N))
    k2[0,2,2,0]=U0   # left QD
    k2[1,3,3,1]=U0    # right QD
    op = InteractionOperator(0.,kappa, k2)  # 2nd quantization operator
    Hmatrix = get_sparse_operator(op).todense()  # get H matrix
    ### use openFermion and qml to convert to Pauli string
    f_op=get_fermion_operator(op)  # get Fermion operator
    f_jw=jordan_wigner(f_op)  # Jordan-Wigner mapping, openFermion
    qml_jw=qchem.convert_observable(f_jw) # from OpenFermion to pennylane
    f_bk=bravyi_kitaev(f_op)  # Bravyi-Kitaev mapping
    qml_bk=qchem.convert_observable(f_bk) # from OpenVermion to pennylane
    
    basis=['1100','1001','0110','0011','1010','0101']  # sub H matrix
    for i, b in enumerate(basis): 
        basis[i]=int(b,2)
    Hbasis=Hmatrix[np.ix_(basis,basis)]
    
    return qml_jw,f_jw,qml_bk,f_bk, Hmatrix,Hbasis,kappa,k2

# mapping to Pauli string Hamiltonian for 4 qubits
def set_tc(scheme,N,ktc,tc, tc0=0.01):
    if scheme==1:   # all bonds uniform tc
        Htc=tc*(ktc+ktc.T)
    else:  # middle bond tc, otherwise tc0
        Htc=tc0*(ktc+ktc.T)
        ind=2  # bond index modulated
        Nh=int(N/2)
        Htc[ind-1,ind]=tc
        Htc[ind,ind-1]=tc
        Htc[ind-1+Nh,ind+Nh]=-tc
        Htc[ind+Nh,ind-1+Nh]=-tc
    return Htc

def getH4(Ez=[1.,0.9,1.,0.9],Udet=1.0, tc=0.05,U0=1.2,scheme=1):  
    N=8
    Nq=int(N/2)    
    zm=np.array([1,-1])
    kd=np.kron(zm,np.array(Ez)) # Zeeman
    kd+=(Udet/2)*np.kron(np.ones(2),np.array([1,-1,1,-1]))  # electostatic potential
    ktc=np.zeros((N,N)) # 1-body term
    for i in range(Nq-1):
        ktc[i,i+1]=1.0
        ktc[i+Nq,i+Nq+1]=-1.0
    Htc=set_tc(scheme,N,ktc,tc)  # Hamiltonian for tunnel coupling
    kappa=Htc+np.diag(kd)

    k2=np.zeros((N, N, N, N))
    for ii in range(Nq): 
        k2[ii,ii+Nq,ii+Nq,ii]=U0
    op = InteractionOperator(0.,kappa, k2)  # 2nd quantization operator
    Hmatrix = get_sparse_operator(op).todense()  # get H matrix
    f_op=get_fermion_operator(op)  # get Fermion operator
    f_jw=jordan_wigner(f_op)  # Jordan-Wigner mapping
    qml_jw=qchem.convert_observable(f_jw) # from OpenFermion to pennylane
    f_bk=bravyi_kitaev(f_op)  # Bravyi-Kitaev mapping
    qml_bk=qchem.convert_observable(f_bk) # from OpenVermion to pennylane
    
    basis=['01011010','10011010','00111010','01101010',
           '01010110','01011100','01011001']  # sub H matrix
    for i, b in enumerate(basis): 
        basis[i]=int(b,2)
    Hbasis=Hmatrix[np.ix_(basis,basis)]
    return qml_jw,f_jw,qml_bk,f_bk, Hmatrix,Hbasis,kappa,k2

def write_pauli_string(qml_op, filename="pauli_test.txt",Nq=4):
    #  qml_op: pauli string from qml
    #  Nq: number of qubits
    import re
    coefs=qml_op.terms[0]
    string=qml_op.terms[1]
    with open(filename, "w") as file:
        for ii, coef in enumerate(coefs):
            pauli=['I']*Nq
            plist=str(string[ii]).split('@')
            for _, pterm in enumerate(plist):
                qind=int(pterm.split("[",1)[1].split("]",1)[0])  # character betwween []
                if pterm.find('Z')!=-1:
                    gate='Z'
                elif pterm.find('X')!=-1:
                    gate='X'
                elif pterm.find('Y')!=-1:
                    gate='Y'
                else:
                    gate='I'
                pauli[qind]=gate              
            file.write(''.join(pauli)+'   ('+str(coef)+')\n')
 
if __name__ == '__main__':
    Nq=4  # number of modedled qubits*2
    if Nq==4:
        qml_jw,f_jw,qml_bk,f_bk, Hmatrix,Hbasis,H1body,H2body=getH()
        write_pauli_string(qml_jw,filename="DQD_jw.txt",Nq=Nq)
        write_pauli_string(qml_bk,filename="DQD_bk.txt",Nq=Nq)
    elif Nq==8:
        qml_jw,f_jw,qml_bk,f_bk, Hmatrix,Hbasis=getH4()
        write_pauli_string(qml_jw,filename="Q4_jw.txt",Nq=Nq)
        write_pauli_string(qml_bk,filename="Q4_bk.txt",Nq=Nq)
    else:
        print('only convert 2 and 4 quantum dots')
    