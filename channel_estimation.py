import numpy as np
import random

rx_antenna_count = 1
tx_antenna_count = 20

h = [0.4, 0.2, 0.6, 0.7, 0.8]

N = tx_antenna_count
M = len(h)
noise_var = 1


if(N < M):
    print("Insufficent Pilot")
    exit()

pilot = np.random.randint(1, 20, size=(N, M))

# Least Square parameter estimation
X = pilot
H = np.array(h)
V = np.random.normal(0, np.sqrt(noise_var), N)
Y = X@H.T + V

H_hat = np.linalg.inv(X.T@X)@X.T@Y
MSE_H_hat = np.sum((H - H_hat)**2)
# print(H)
# print(H_hat)
# print(f"Error in estimate is : {MSE_H_hat}")

# MIMO Least Square parameter estimation
rx_antenna_count = 10
tx_antenna_count = 20
noise_var = 1
h_var = 2
h_mean = 10

r = 3
t = 4
N = 5

X = np.random.randint(1, 100, size=(t, N))
H = np.random.normal(h_mean, np.sqrt(h_var), (r, t))
V = np.random.normal(0, np.sqrt(noise_var), (r, N))

Y = H@X + V

H_hat = Y@X.T@np.linalg.inv(X@X.T)
MSE_H_hat = np.sum(np.sum((H - H_hat)**2))

print(H)
print(H_hat)
print(f"Error in estimate is : {MSE_H_hat}")