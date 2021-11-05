import numpy as np
print(np.pi/2*np.ones(6))

print("Initial parameters (example 1.57079633,1.57079633,1.57079633,1.57079633,1.57079633,1.57079633): ")
init_params = input()
init_params = [float(x) for x in init_params.split(',')]

print(init_params)