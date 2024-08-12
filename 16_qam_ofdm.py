# 16 QAM using OFDM

import numpy as np
import scipy.interpolate
import scipy
import matplotlib.pyplot as plt

###############################################################
#                     Setup                                   #
###############################################################


K = 64      # number of OFDM subcarriers
CP = K//4   # length of the cyclic prefix: 25% of the block
P = CP//2   # number of pilot carriers per OFDM block
pilotValue = 3+3j # The known value each pilot transmits

# maping table 
mapping_table = {
(0,0,0,0) : -3-3j,
(0,0,0,1) : -3-1j,
(0,0,1,0) : -3+3j,
(0,0,1,1) : -3+1j,
(0,1,0,0) : -1-3j,
(0,1,0,1) : -1-1j,
(0,1,1,0) : -1+3j,
(0,1,1,1) : -1+1j,
(1,0,0,0) :  3-3j,
(1,0,0,1) :  3-1j,
(1,0,1,0) :  3+3j,
(1,0,1,1) :  3+1j,
(1,1,0,0) :  1-3j,
(1,1,0,1) :  1-1j,
(1,1,1,0) :  1+3j,
(1,1,1,1) :  1+1j
}

demapping_table = {v : k for k, v in mapping_table.items()}

# Reshape bit for maping them to symbols
def SP(bits):
  return bits.reshape((len(dataCarriers), mu))

# maping bits block to symbol
def Mapping(bits):
  return np.array([mapping_table[tuple(b)] for b in bits])

# generate ofdm block with pilot and payload
def OFDM_symbol(QAM_payload):
  symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
  symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
  symbol[dataCarriers] = QAM_payload  # allocate the pilot subcarriers
  return symbol

# IFFT of data
def IFFT(OFDM_data):
  return np.fft.ifft(OFDM_data)

# FFT of data
def FFT(OFDM_RX):
  return np.fft.fft(OFDM_RX)

# add cyclic prefix to data
def addCP(OFDM_time):
  cp = OFDM_time[-CP:]               # take the last CP samples ...
  return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

# remove cyclic prefix
def removeCP(signal):
  return signal[CP:(CP+K)]


###############################################################
#                     Transmitter                             #
###############################################################

allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.
# For convenience of channel estimation, let's make the last carriers also be a pilot
pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
P = P+1

# data carriers are all remaining carriers
dataCarriers = np.delete(allCarriers, pilotCarriers)

# print ("allCarriers:   %s\n" % allCarriers)
# print ("pilotCarriers: %s\n" % pilotCarriers)
# print ("dataCarriers:  %s\n" % dataCarriers)

mu = 4 # bits per symbol (i.e. 16QAM)

# number of payload bits per OFDM symbol
payloadBits_per_OFDM = len(dataCarriers)*mu  

# data to be send.
bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
bits_SP = SP(bits)
# print ("16 bit groups")
# print (bits_SP[:16,:])

print ("Bits count: ", len(bits))
print ("First 20 bits: ", bits[:64])
print ("Mean of bits (should be around 0.5): ", np.mean(bits))

QAM = Mapping(bits_SP)
print ("First 16 QAM symbols and bits:")
# print (bits_SP[:16,:])
# print (QAM[:16])


OFDM_data = OFDM_symbol(QAM)
print ("Number of OFDM carriers in frequency domain: ", len(OFDM_data))

OFDM_time = IFFT(OFDM_data)
print ("Number of OFDM samples in time-domain before CP: ", len(OFDM_time))

OFDM_withCP = addCP(OFDM_time)
print ("Number of OFDM samples in time domain with CP: ", len(OFDM_withCP))

OFDM_TX = OFDM_withCP


###############################################################
#                     Channel                                 #
###############################################################

# the impulse response of the wireless channel
channelResponse = np.array([1, 0, 0.3+0.3j])  

# computes the one-dimensional n-point DFT
# calculating channel frequency response
H_exact = np.fft.fft(channelResponse, K)
plt.plot(allCarriers, abs(H_exact),'.-',label='Channel frequency Response')
plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)

SNRdb = -10  # signal to noise-ratio in dB at the receiver 


def channel(signal):
  convolved = np.convolve(signal, channelResponse)
  signal_power = np.mean(abs(convolved**2))
  sigma2 = signal_power * 10**(-SNRdb/10)  # calculate noise power based on signal power and SNR
  print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
  # Generate complex noise with given variance
  noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
  return convolved + noise



###############################################################
#                     Reciever                                #
###############################################################

OFDM_RX = channel(OFDM_TX)
#plt.figure(figsize=(15,4))
#plt.plot(abs(OFDM_TX), label='TX signal', color='green')
#plt.plot(abs(OFDM_RX), label='RX signal',color='red')
#plt.legend(fontsize=10)
#plt.xlabel('Time'); plt.ylabel('$|x(t)|$');
#plt.grid(True);

OFDM_RX_noCP = removeCP(OFDM_RX)
OFDM_demod = FFT(OFDM_RX_noCP)

def channelEstimate(OFDM_demod):
  pilots = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal
  Hest_at_pilots = pilots / pilotValue # divide by the transmitted pilot values

  # Perform interpolation between the pilot carriers to get an estimate
  # of the channel in the data carriers. Here, we interpolate absolute value and phase 
  # separately
  Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
  Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
  Hest = Hest_abs * np.exp(1j*Hest_phase)

  #plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
  #plt.stem(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
  #plt.plot(allCarriers, abs(Hest), label='Estimated channel via interpolation')
  #plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
  #plt.ylim(0,2)
  return Hest

Hest = channelEstimate(OFDM_demod)

#print(Hest)

def equalize(OFDM_demod, Hest):
  return OFDM_demod / Hest
equalized_Hest = equalize(OFDM_demod, Hest)

def get_payload(equalized):
  return equalized[dataCarriers]


#QAM_est_before_equ = get_payload(OFDM_demod)
#plt.plot(QAM_est_before_equ.real, QAM_est_before_equ.imag, 'ro',label='Before Equalization');
QAM_est = get_payload(equalized_Hest)
#plt.plot(QAM_est.real, QAM_est.imag, 'bo',label='After Equalization');
#plt.legend(fontsize=8)

def Demapping(QAM):
  # array of possible constellation points
  constellation = np.array([x for x in demapping_table.keys()])
  # calculate distance of each RX point to each possible point
  dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
  # for each element in QAM, choose the index in constellation 
  # that belongs to the nearest constellation point
  const_index = dists.argmin(axis=1)
  # get back the real constellation point
  hardDecision = constellation[const_index]
  # transform the constellation point into the bit groups
  return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision

PS_est, hardDecision = Demapping(QAM_est)
#for qam, hard in zip(QAM_est, hardDecision):
  #plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o');
  #plt.plot(hardDecision.real, hardDecision.imag, 'go')
  #plt.ylim(-7)

#print(bits_est)
def PS(bits):
  return bits.reshape((-1,))

bits_est = PS(PS_est)
#print(bits_est)
print ("Obtained Bit error rate: ", np.sum(abs(bits-bits_est))/len(bits))