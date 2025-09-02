import numpy as np
SNR_dB = np.linspace(-5,5,21)

sigma_n2 = 10**(-SNR_dB/10)
sigma_s2 = 1

angle = 0
K = 16

beamgain = 1
N_win = 1 #np.linspace(1,15,15)

CRB = sigma_n2/(np.pi**2 * np.cos(angle)**2 *2*N_win) * (beamgain*sigma_s2*((sigma_n2*K*beamgain*sigma_s2)/(sigma_n2*(sigma_n2+K*beamgain*sigma_s2)))*(0.5*K**3-0.5*K)/6)**(-1)

print(CRB)

CRB_sqrt = np.sqrt(CRB)

print(CRB_sqrt)
