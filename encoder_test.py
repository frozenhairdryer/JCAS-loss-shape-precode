from Autoencoder.functions import QAM_encoder, QAM_decoder,rayleigh_channel,BER
import torch
import numpy as np
import matplotlib.pyplot as plt

M=torch.tensor([16])
symbols = torch.randint(0,int(M),(100,))
noise=torch.tensor([0.1]).repeat(100)
CSI=torch.tensor([1+1j]).repeat(100)

enc = QAM_encoder(M)
encoded = enc(symbols)
encoded_code = enc.coding()[symbols]
channel, beta = rayleigh_channel(encoded,1,noise,1)
CSI = beta
# no channel

dec = QAM_decoder(M)
decoded, sym_out = dec(channel,CSI,noise)
print("BER is ", float(torch.mean(BER(decoded,encoded_code,M))))
