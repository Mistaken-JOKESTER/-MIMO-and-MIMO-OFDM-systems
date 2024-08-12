import numpy as np
import matplotlib.pyplot as plt
import random


def ofdm_tx(N, data_bits):
  # Generate BPSK symbols
  bpsk_symbols = 2 * data_bits - 1

  # Create IDFT matrix
  idft_mat = np.fft.ifft(np.eye(N))

  # OFDM modulation
  ofdm_symbols = np.dot(idft_mat, bpsk_symbols)

  # Add cyclic prefix (assume CP length is N/4)
  cp_length = N // 4
  cp = ofdm_symbols[-cp_length:]
  ofdm_tx_signal = np.concatenate((cp, ofdm_symbols))

  return ofdm_tx_signal


def channel(ofdm_tx_signal, h):
  ofdm_rx_signal = np.convolve(ofdm_tx_signal, h)
  return ofdm_rx_signal


def ofdm_rx(ofdm_rx_signal, N, h):
  # Remove cyclic prefix
  cp_length = N // 4
  ofdm_rx_signal = ofdm_rx_signal[cp_length:]

  # Perform DFT
  ofdm_freq = np.fft.fft(ofdm_rx_signal)

  # Channel frequency response
  H = np.fft.fft(h, N)

  # Channel equalization
  equalized = ofdm_freq / H

  # Obtain BPSK symbols
  bpsk_rx_symbols = equalized[:N]

  return bpsk_rx_symbols


def main():
  N = 16  # Number of subcarriers
  data_bits = np.random.randint(2, size=N)
  h = np.array([1, 0.5, 0.3])

  ofdm_tx_signal = ofdm_tx(N, data_bits)
  ofdm_rx_signal = channel(ofdm_tx_signal, h)
  bpsk_rx_symbols = ofdm_rx(ofdm_rx_signal, N, h)

  # You can add code here for bit error rate calculation, visualization, etc.

def generate_random_bits(num_bits):
  """Generates a list of random bits.

  Args:
    num_bits: The number of random bits to generate.

  Returns:
    A list of random bits (0 or 1).
  """

  random_int = random.getrandbits(num_bits)
  random_bits = [int(x) for x in bin(random_int)[2:]]
  return random_bits


num_bits = 10**3
random_bits = generate_random_bits(num_bits)
print(random_bits)