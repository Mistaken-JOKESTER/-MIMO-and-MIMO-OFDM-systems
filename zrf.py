

# Complete Interference Cancellation: ZF perfectly eliminates
# interference between data streams.

# Noise Amplification: Due to the matrix inversion, 
# noise is amplified, leading to performance degradation 
# compared to other techniques like Minimum Mean Square Error (MMSE).

# Feasibility: Requires the number of receive antennas to be greater 
# than or equal to the number of transmit antennas.


# Mathematical Representation
    # Let's denote:

    # H: Channel matrix (NxM, where N is the number of 
    # receive antennas and M is the number of transmit antennas)

    # x: Transmitted data vector (Mx1)
    # y: Received signal vector (Nx1)
    # n: Noise vector (Nx1)
    # The MIMO system can be modeled as:

    # y = Hx + n
    # The ZF receiver is given by:

    # W = (H^H * H)^-1 * H^H
    # x_hat = W * y
    # Where:

    # W: ZF equalizer matrix
    # H^H: Hermitian transpose of H


# Performance Considerations
    # While ZF offers complete interference cancellation, 
    # it suffers from noise amplification, especially when the 
    # condition number of the channel matrix is high. This leads 
    # to performance degradation in terms of Bit Error Rate (BER).

# Condition Number: In numerical analysis, the condition number of a 
# function measures how much the output value of the function can 
# change for a small change in the input argument

# To mitigate this:

    # Regularization: Adding a small diagonal 
    # matrix to H^H * H before inversion can improve stability.
    # 
    # Other Techniques: MMSE receivers offer a better trade-off 
    # 
    # between interference cancellation and noise amplification.


import numpy as np
import scipy.linalg as la

def generate_random_x(M):
  """
  Generates a random complex vector of size M.

  Args:
    M: Number of transmit antennas

  Returns:
    x: Randomly generated transmit vector
  """

  x = np.random.randn(M) + 1j * np.random.randn(M)
  return x

def add_noise(y, noise_var):
  """
  Adds AWGN noise to the received signal.

  Args:
    y: Received signal vector
    noise_var: Noise variance

  Returns:
    y_noisy: Noisy received signal vector
  """

  noise = np.sqrt(noise_var/2) * (np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape))
  y_noisy = y + noise
  return y_noisy

def zero_forcing_receiver(H, y):
  """
  Implements a Zero Forcing receiver for a MIMO system.

  Args:
    H: Channel matrix (numpy array)
    y: Received signal vector (numpy array)

  Returns:
    x_hat: Estimated transmitted signal vector (numpy array)
  """

  # Calculate the pseudo-inverse of the channel matrix
  W = la.pinv(H)

  # Apply the ZF equalizer to the received signal
  x_hat = np.dot(W, y)

  return x_hat


def simulate_mimo_system(H, noise_var):
  """
  Simulates a MIMO system with ZF receiver.

  Args:
    H: Channel matrix
    noise_var: Noise variance

  Returns:
    x_hat: Estimated transmitted signal
  """

  M = H.shape[1]  # Number of transmit antennas
  x = generate_random_x(M)
  y = np.dot(H, x)
  y_noisy = add_noise(y, noise_var)
  x_hat = zero_forcing_receiver(H, y_noisy)
  return x_hat, x

# Example usage
r = 4
t = 2
H = np.random.randn(r, t) + 1j * np.random.randn(r, t)
noise_var = 0.1
x_hat, x = simulate_mimo_system(H, noise_var)

print("Estimated x:", x_hat)
print("Original x:", x)