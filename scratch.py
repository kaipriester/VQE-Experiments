#----------------------------------------------------------------------------------------------------------------------

def optimizer(n, define_VQE_ansatz, starting_point, total_test,
             param=[0.,0.], simulated_noise=False, measure_list=[], pauli_list=[], define_overlap=None):
    weight=cal_weight(n,pauli_list)
    exact_E, Hfull=E_from_numpy(n,pauli_list)
    L = len(param)
    F=[]
    def objective_function(x):
        E=-0.10312
    return E 
    
    for T in range(starting_point, total_test):
        #for F_plus
        for i in range(L):
            param[i]= minimize(objective_function, F[:,0][i], bounds = np.array([[-np.pi, np.pi], [-np.pi, np.pi]], dtype=float) , budget = 1, method='imfil')
        list_dict=[]
        for _,mm in enumerate(measure_list):
            list_dict.append(get_VQE_result(n, param, simulated_noise,mm,define_VQE_ansatz))
        if define_overlap is None:
            F_plus = evaluate(list_dict,weight)

        #for F_minus
        for i in range(L):
            param[i]= minimize(objective_function, F[:,0][i], bounds = np.array([[-np.pi, np.pi], [-np.pi, np.pi]], dtype=float) , budget = 1, method='imfil')
        list_dict=[]
        for _,mm in enumerate(measure_list):
            list_dict.append(get_VQE_result(n, param,  simulated_noise,mm,define_VQE_ansatz))
        if define_overlap is None:
            F_minus = evaluate(list_dict,weight)    

        ### calculate F value
        list_dict=[]
        for _,mm in enumerate(measure_list):
            list_dict.append(get_VQE_result(n, param, simulated_noise,mm,define_VQE_ansatz))
        E_middle = evaluate(list_dict,weight)
        
        E_middle_exact = float(evaluate_with_Hamiltonian(n, param, Hfull,define_VQE_ansatz))
 
        #store the parameters in the current step
        F.append([E_middle,E_middle_exact])    
    return F, exact_E

def run_opt(define_VQE_ansatz, Nparam, measure_list, pauli_list, define_overlap=None):
    F, exact_E = optimizer(12, define_VQE_ansatz, 0, 30, param=np.zeros(Nparam), simulated_noise=True, measure_list=measure_list, pauli_list=pauli_list, define_overlap=define_overlap)     
    #checking results
    print("F: ", type(F), F) 
    print("exact_E: ", type(exact_E), exact_E)

    ### plot results
    #exact_E=-0.1026  # need to edit for every case
    #should converge to exact_E
    def plot_result(F, exact_E):
        Nitr=len(F)
        plt.figure()
        E = np.full((Nitr),exact_E[3])
        F = np.array(F)[:,0]
        xv=np.arange(Nitr)+1
        plt.plot(xv, F) #plot just the first column of F
        plt.plot(xv, E,'g--')
        plt.xlabel('iteration number')
        plt.ylabel('energy')
        plt.show()
    plot_result(F,exact_E) 
#-----------------------------------------------------------------------------------------------------------------------