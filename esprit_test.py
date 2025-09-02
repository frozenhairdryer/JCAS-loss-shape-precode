#from Autoencoder.NN_classes import PSK_encoder
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from Autoencoder.functions import esprit_angle,esprit_angle_nns, matrix_pencil_method, radar_channel_swerling1,radiate

#r = torch.zeros((16*4,20))-1/np.sqrt(64)+0j
f=100
#received = torch.zeros((16*4,20))
t = torch.linspace(0,0.5,10)
for t1 in t:
    mpm=[]
    r = torch.zeros((f,16*1))+torch.ones((f,16))*0.5*radiate(16,t1,1)+torch.rand((f,16*1))*0.5
    #r = torch.ones(100,16)+0j+torch.rand((100,16*1))*0.2 
    #r,_ = radar_channel_swerling1(r,0.1,0.1,0.1,torch.tensor([16,1]), torch.tensor([[[t1], [np.pi/2]]]))
    #received += r
    esprit_1 = esprit_angle(torch.squeeze(r).detach().cpu().numpy(),torch.tensor([16,1]),1,f)
    esprit_2 = esprit_angle_nns(r,torch.tensor([16,1]),torch.tensor([1]),torch.tensor([f]))
    for a in range(f):
        mpm.append(matrix_pencil_method(r[a],torch.tensor([16,1]),torch.tensor([1])).detach().cpu().numpy())
    mpml = np.mean(mpm)
    print("Input angle is:" + str(t1))
    print("Esprit conventional: "+str(esprit_1))
    print("Esprit NNs: "+str(esprit_2))
    print("Matrix Pencil Method: "+str(mpml))



