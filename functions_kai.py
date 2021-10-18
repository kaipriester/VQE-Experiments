from sklearn.model_selection import GridSearchCV
import pennylane as qml
from noisyopt import minimizeSPSA
import numpy as np
import qiskit as qk
from sklearn.model_selection import train_test_split
import inspect

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
 
def define_ansatz(param, n, list_1= [1,2]):
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

def get_qc(params, n):
    list_1= [1,2]  # exited state is 1001 in Lu, Ru, Ld, Rd
    qc,qr,cr=define_ansatz(params, n, list_1)
    return qc
   
    # init_params = qml.init.strong_ent_layers_normal(
    # n_wires=n_wires, n_layers=n_layers
    # )
    #params = init_params.copy()
    # np.random.seed(0)
    # init_params = np.random.normal(0, np.pi, (n_wires, 3))
    # params = init_params.copy()
    # print(params)
    # print(type(params))  

def GD(steps=30, n_wires=4, n_layers=6, stepsize=0.3, exact_E = -0.1026, device = qml.device("default.qubit", wires=4)):
    #USE pennylane's GradientDescentOptimizer PACKAGE TO CALCULATE ENERGIES
    init_params = np.zeros(n_layers)
    
    circuit_qk = get_qc(init_params, n_wires)
    circuit_qml = qml.from_qiskit(circuit_qk)
    print("circuit_qml: ")
    print(type(circuit_qml))
    print(inspect.signature(circuit_qml))

    def circuit(params_2):
        circuit_qml(params_2, range(device.num_wires))
        return exact_E

    # all_pauliz_tensor_prod = qml.operation.Tensor(*[qml.PauliZ(i) for i in range(n_wires)])
    # def circuit(params):
    #     qml.templates.StronglyEntanglingLayers(params, wires=list(range(n_wires)))
    #     return qml.expval(all_pauliz_tensor_prod)

    # def my_quantum_function(x):
    #     qml.RZ(x, wires=0)
    #     qml.CNOT(wires=[0,1])
    #     #qml.RY(y, wires=1)
    #     return qml.expval(qml.PauliZ(1))

    opt = qml.GradientDescentOptimizer(stepsize=stepsize)    
    # Parameters
    #     func (callable) – a quantum function
    #     device (Device) – a PennyLane-compatible device    
    cost = qml.QNode(circuit, device)

    params_1 = init_params.copy()

    for k in range(steps):
        print("step:  " + str(k))
        params1, energy = opt.step_and_cost(cost, params_1)
        print("energy at step " + str(k) +": " + str(energy)) 
        print("params at step " + str(k) +": " + str(params1)) 

    #return energy

def SPSA(steps=30, n_wires=4, n_layers=6, c=0.3, a=1.5, exact_E = -0.1026, device = qml.device("default.qubit", wires=4)):


    #USE noisyopt's minimizeSPSA PACKAGE TO CALCULATE ENERGIES
    flat_shape = n_layers * n_wires * 3
    init_params = qml.init.strong_ent_layers_normal(
        n_wires=n_wires, n_layers=n_layers
    )
    init_params_spsa = init_params.reshape(flat_shape)

    # circuit_qk = get_qc(init_params_spsa.copy(), n_wires)
    # circuit_qml = qml.from_qiskit(circuit_qk)

    # def circuit(params):
    #     circuit_qml(params, range(device.num_wires))
    #     return exact_E

    all_pauliz_tensor_prod = qml.operation.Tensor(*[qml.PauliZ(i) for i in range(n_wires)])
    def circuit(params):
        qml.templates.StronglyEntanglingLayers(params, wires=list(range(n_wires)))
        return qml.expval(all_pauliz_tensor_prod)

    qnode_spsa = qml.QNode(circuit, device)

    def cost_spsa(params):
        return qnode_spsa(params.reshape(n_layers, n_wires, 3))

    cost_store_spsa = [cost_spsa(init_params_spsa)]
    device_execs_spsa = [0]

    def callback_fn(xk):
        cost_val = cost_spsa(xk)
        cost_store_spsa.append(cost_val)

    niter_spsa = 200

    # Evaluate the initial cost
    cost_store_spsa = [cost_spsa(init_params_spsa)]

    res = minimizeSPSA(
        cost_spsa,
        x0=init_params_spsa.copy(),
        niter=niter_spsa,
        paired=False,
        c=c,
        a=a,
        callback=callback_fn,
    )

    print("energy values")
    print(res.x)
    #return res  

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

    params = np.zeros(n_layers)
    circuit_qk = get_qc(params, n_wires)
    circuit_qml = qml.from_qiskit(circuit_qk)

    def circuit(params):
        circuit_qml(params, range(device.num_wires))
        return exact_E


    cost_spsa = qml.QNode(circuit, device)

    #init_params = np.random.normal(0, np.pi, (n_wires, 3))
    #params = init_params.copy().reshape(n_wires * n_layers)

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

if __name__ == '__main__':
    #hyperpara_opt_spsa()
    #hyperpara_opt_gd()
    SPSA()
    
    #GD()