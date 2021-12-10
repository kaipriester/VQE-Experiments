# Engineering Undergraduate Research (EGN 4912)
Student: Kai Priester

Prof: Jing Guo

Mentor: Ning Yang

Topic: Variational Hybrid Quantum-Classical Algorithms 

Semester: Fall 2021

## CLI for Analyzing Optimizers in Variational Hybrid Quantum-Classical Algorithms

* This CLI (QMLCirOpts_GD_SPSA.py) allows the user to run different optimizers on the quantum circuit defined in (pennyl3.py) 
* First you can choose the optimizer(s) you want to run… as it runs each it will give you the choice of using default parameters or editing them… at the end of each optimizer it will display the energy per step graph (also saves this to results folder) and the execution time

### Optimizers Available:

* Basic gradient-descent optimizer (GD)
* Simultaneous Perturbation Stochastic Approximation (SPSA)
* Gradient-descent optimizer with adaptive learning rate, first and second moment (Adam)
* Gradient-descent optimizer with past-gradient-dependent learning rate in each dimension (Ada)
* Gradient-descent optimizer with momentum (MO)
* Gradient-descent optimizer with Nesterov momentum (NMO)
* Root mean squared propagation optimizer (RMSProp)
* Minimization of an objective function by a pattern search (MinCompass)

### How to Run
Have python3.6+ and pip installed
1. Setup the [Anaconda Prompt](https://docs.conda.io/en/latest/miniconda.html)
2. Clone repo 

    `git clone https://github.com/kaipriester/VQE-Experiments.git`
3. Create conda virtual environment

    `conda create --name <env_name>`

    `conda activate <env_name>`    
4.  Install any packages needed

    `pip install ...`
5. Run CLI script

    `python QMLCirOpts_GD_SPSA.py`