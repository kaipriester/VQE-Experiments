def optimizer2(n, starting_point, total_test, Nparam, measure_list, pauli_list, define_VQE_ansatz, simulated_noise=False):
    weight=cal_weight(n,pauli_list)

    #USING OTHER DEVICES TO AVIOD ERRORS
    #dev_sampler_gd = qml.device("qiskit.aer", wires=Nparam, shots=1000)
    dev = qml.device('default.qubit', wires=Nparam)

    # Initialize the optimizer - optimal step size was found through a grid search
    #UPDATE STEP SIZE
    opt = qml.GradientDescentOptimizer(stepsize=2.2)
    print(pauli_list)
    #I THINK pauli_list IS THE HAMILTONIAN BUT IT IS IN THE WRONG FORMAT
    #DEF ExpvalCost(ansatz, hamiltonian, device, interface='autograd', diff_method='best', optimize=False, **kwargs)
    
    pauli_to_hamil(pauli_list)
    H = pauli_to_hamil(pauli_list)

    
    def circuit(params, wires):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
        for i in wires:
            qml.Rot(*params[i], wires=i)
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[2, 0])
        qml.CNOT(wires=[3, 1])  
    
    def exp_val_circuit(params):
        circuit(params, range(dev.num_wires))
        return qml.expval(H)
  

    #NOW IS SAYING device_Q IS NOT RIGHT 
    #AttributeError: 'QasmSimulator' object has no attribute 'wires'
    #EITHER ONE OF THESE
    ##device_16 = provider.get_backend('ibmq_16_melbourne')
    #device_Q=qk.Aer.get_backend('qasm_simulator')
    #NEEDS TO BE https://pennylane.readthedocs.io/en/stable/code/api/pennylane.device.html#pennylane.device
    #cost = qml.ExpvalCost(ansatz, H, dev)
    cost = qml.QNode(exp_val_circuit, dev)

    # This random seed was used in the original VQE demo and is known to allow the
    # algorithm to converge to the global minimum.
    np.random.seed(0)
    init_params = np.random.normal(0, np.pi, (n, 3))
    params = init_params.copy()
    energies = []

    print(total_test)
    # Run the gradient descent algorithm
    for n in range(total_test):
        print("in gd")
        #IS FAILING ON THIS LINE NOW
        params, energy = opt.step_and_cost(cost, params)

        energies.append(energy)

        if n % 5 == 0:
            print(
                f"Iteration = {n}, "
                f"Number of device executions = {device_Q.num_executions},  "
                f"Energy = {energy:.8f} Ha"
            )

    return energies