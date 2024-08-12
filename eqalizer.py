import numpy as np
import random

rx_antenna_count = 1
tx_antenna_count = 20

h = [0.4, 0.2, 0.6, 0.7, 0.8]

N = tx_antenna_count
M = len(h)
r = tx_antenna_count

noise_var = 1
h_var = 3
h_mean = 2

if(N < M):
    print("Insufficent Pilot")
    exit()

data = np.random.randint(1, 20, size=(N, 30))

# Least Square parameter estimation
X = data
H = np.random.normal(3, np.sqrt(noise_var), size=(N,N))
V = np.random.normal(0, np.sqrt(noise_var), size=(N, 30))
Y = H@X + V

X_hat = np.linalg.inv(H.T@H)@H.T@Y
MSE_X_hat = np.sum((X - X_hat)**2)

print(f"Error in estimate is : {MSE_X_hat}")