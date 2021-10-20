from sklearn.model_selection import GridSearchCV
import pennylane as qml
from noisyopt import minimizeSPSA
import numpy as np
import qiskit as qk
from sklearn.model_selection import train_test_split
import inspect
import matplotlib.pyplot as plt 
import time

from test_DQDjw_vqed import *

# default cicuit
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
 
def define_ansatz(param2, n, list_1= [1,2]):
    qca, p=A_ij()
    
    ### quantum circuit for ansatz
    qr = qk.QuantumRegister(n, 'q')
    cr = qk.ClassicalRegister(n, 'c')
    qc = qk.QuantumCircuit(qr,cr)
    ### initialize the ansatz state
    for  ix in list_1:
        qc.x(qr[ix])
    param2=np.array(param2)+np.pi/2
    sub_ck=[qca.bind_parameters({p: p_val}) for p_val in param2]
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

def get_qc(params1, n):
    list_1= [1,2]  # exited state is 1001 in Lu, Ru, Ld, Rd
    qc,qr,cr=define_ansatz(params1, n, list_1)
    return qc
   
def GD(steps=200, n_wires=4, n_layers=6, stepsize=0.3, exact_E = -0.1026, device = qml.device("default.qubit", wires=4)):
    #USE pennylane's GradientDescentOptimizer PACKAGE TO CALCULATE ENERGIES

    qc = get_qc(np.zeros(n_layers), n_wires)
    def circuit(x):
        qml.from_qiskit(qc)
        #NEED TO WORK ON THIS LINE
        #return qml.expval(qml.PauliZ(0))

    # all_pauliz_tensor_prod = qml.operation.Tensor(*[qml.PauliZ(i) for i in range(n_wires)])
    # def circuit(params):
    #     qml.templates.StronglyEntanglingLayers(params, wires=list(range(n_wires)))
    #     return qml.expval(all_pauliz_tensor_prod)

    opt = qml.GradientDescentOptimizer(stepsize=stepsize)    
    cost = qml.QNode(circuit, device)

    flat_shape = n_layers * n_wires * 3
    params = qml.init.strong_ent_layers_normal(
        n_wires=n_wires, n_layers=n_layers
    )

    F = []
    for k in range(steps):
        params, energy = opt.step_and_cost(cost, params)
        F.append(energy)
        print("energy at step " + str(k) +": " + str(energy)) 

    return F

def SPSA(steps=100, n_wires=4, n_layers=6, c=0.3, a=1.5, exact_E = -0.1026, device = qml.device("default.qubit", wires=4)):
    #USE noisyopt's minimizeSPSA PACKAGE TO CALCULATE ENERGIES
    all_pauliz_tensor_prod = qml.operation.Tensor(*[qml.PauliZ(i) for i in range(n_wires)])
    def circuit(params):
        qml.templates.StronglyEntanglingLayers(params, wires=list(range(n_wires)))
        return qml.expval(all_pauliz_tensor_prod)

    qnode_spsa = qml.QNode(circuit, device)

    def cost_spsa(params):
        return qnode_spsa(params.reshape(n_layers, n_wires, 3))

    flat_shape = n_layers * n_wires * 3
    init_params = qml.init.strong_ent_layers_normal(
        n_wires=n_wires, n_layers=n_layers
    )
    init_params_spsa = init_params.reshape(flat_shape)

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
        niter=200,
        paired=False,
        c=c,
        a=a,
        callback=callback_fn,
    )

    print("The solution of the optimization:")
    print(res.x)
    print("Whether or not the optimizer exited successfully:")
    print(res.success)
    # print("Termination status of the optimizer. Its value depends on the underlying solver:")
    # print(res.status)
    print("Description of the cause of the termination.")
    print(res.message)
    return F

#MAKE GRIDESEARCH FUNCTION
def hyperpara_opt_gd():
    F,Params,opt_param, opt_E, exact_E, Hmatrix, Hbasis = sim_one()
    X = opt_param
    exact_E = -0.1026
    y = np.full((1,6), exact_E).flatten()
    print(y)
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    param_grid = {'stepsize': [0.001, 0.01, 0.1, 0.5, 1., 1.5, 2., 2.5, 3., 3.5]}
    model = GridSearchCV(qml.GradientDescentOptimizer(), param_grid, cv=5)
    
    model.fit(X_train, y_train)
    model_best = model.best_estimator_

    best_params = model.best_params_
    print(best_params)   

def hyperpara_opt_spsa(steps=30, n_wires=4, n_layers=6, stepsize=0.3, exact_E = -0.1026, device = qml.device("default.qubit", wires=4)):

    all_pauliz_tensor_prod = qml.operation.Tensor(*[qml.PauliZ(i) for i in range(n_wires)])
    def circuit(params):
        qml.templates.StronglyEntanglingLayers(params, wires=list(range(n_wires)))
        return qml.expval(all_pauliz_tensor_prod)

    cost_spsa = qml.QNode(circuit, device)

    params = np.zeros(n_layers)

    # Wrapping the cost function and flattening the parameters to be compatible
    # with noisyopt which assumes a flat array of input parameters
    def wrapped_cost(params):
        return cost_spsa(params.reshape(n_wires, n_layers))

    F,Params,opt_param, opt_E, exact_E, Hmatrix, Hbasis = sim_one()
    X = opt_param
    exact_E = -0.1026
    y = np.full((1,6), exact_E).flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    param_grid = {'c': [0.001, 0.01, 0.1, 0.5, 1., 1.5, 2., 2.5, 3., 3.5],
                'a': [0.001, 0.01, 0.1, 0.5, 1., 1.5, 2., 2.5, 3., 3.5]}
    model = GridSearchCV(minimizeSPSA(wrapped_cost, x0=params), param_grid, cv=5)
    model.fit(X_train, y_train)

    best_params = model.best_params_
    print(best_params)  
    
def plot_result(F, exact_E):
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
    #hyperpara_opt_spsa()
    #hyperpara_opt_gd()
    time_start_SPSA = time.time()
    res1 = SPSA()
    time_stop_SPSA = time.time()
    plot_result(res1, -1.136189454088)   
    time_start_GD = time.time()
    res2 = GD()
    time_stop_GD = time.time()
    plot_result(res2, -1.136189454088)
    print("SPSA time: " + str(time_stop_SPSA - time_start_SPSA))
    print("GD time: " + str(time_stop_GD - time_start_GD))