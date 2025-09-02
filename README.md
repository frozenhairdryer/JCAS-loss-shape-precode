# JCAS-loss-shape-precode
Simulations for paper

# Training
To train a NN instance execute **Autoencoder/run.py**

# Evaluation
To evaluate instances run **autoencoder_SNRsweep_mimo.py**.
You need to adapt the parameters to

- path_list: list paths of system-runner instances
- SNR_s, SNR_c: SNR range of sensing and communication in dB (without beamforming gain)
- enctype: choose NN or QAM for modulation
- num_ue: number of UEs for mimo extension; choose as 1 for single communication signal
