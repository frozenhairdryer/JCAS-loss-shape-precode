from Autoencoder.NN_classes import * #Encoder,Decoder,Radar_receiver,Radar_detect,Angle_est
from Autoencoder.functions import *
from Autoencoder.training_routine_SNRsweep import train_network
#import pickle
#enc =NN_classes.Encoder(M=4)
logging.info("Running autoencoder_compare.py")


M=torch.tensor([16],device=device)
SNR_s = torch.pow(10.0,torch.tensor([-10,10],device=device)/10)
SNR_c = torch.pow(10.0,torch.tensor([5,30],device=device)/10)
wr = 0.5
wname=['001','005','01','015','02','03','04','05','06','07','08','09','095','099']
#methods = ['results/train_NN_lossmod/NN_001.pkl','results/train_NN_lossmod/NN_005.pkl','results/train_NN_lossmod/NN_01.pkl','results/train_NN_lossmod/NN_015.pkl','results/train_NN_lossmod/NN_02.pkl','results/train_NN_lossmod/NN_03.pkl','results/train_NN_lossmod/NN_04.pkl','results/train_NN_lossmod/NN_05.pkl','results/train_NN_lossmod/NN_06.pkl','results/train_NN_lossmod/NN_07.pkl','results/train_NN_lossmod/NN_08.pkl','results/train_NN_lossmod/NN_09.pkl','results/train_NN_lossmod/NN_095.pkl','results/train_NN_lossmod/NN_099.pkl']
methods = ['results/QAM_lossmod/QAM_001.pkl','results/QAM_lossmod/QAM_005.pkl','results/QAM_lossmod/QAM_01.pkl','results/QAM_lossmod/QAM_015.pkl','results/QAM_lossmod/QAM_02.pkl','results/QAM_lossmod/QAM_03.pkl','results/QAM_lossmod/QAM_04.pkl','results/QAM_lossmod/QAM_05.pkl','results/QAM_lossmod/QAM_06.pkl','results/QAM_lossmod/QAM_07.pkl','results/QAM_lossmod/QAM_08.pkl','results/QAM_lossmod/QAM_09.pkl','results/QAM_lossmod/QAM_095.pkl','results/QAM_lossmod/QAM_099.pkl']



for l in range(len(methods)):
    if device=='cuda' and 1==0:
        enc_best, dec_best, beam_best, rad_rec_best = pickle.load( open( methods[l], "rb" ) )
    else:
        enc_best, dec_best, beam_best, rad_rec_best = CPU_Unpickler( open( methods[l], "rb")).load()
    num_targets_trained = rad_rec_best.targetnum
    M = enc_best.M
    enctype=enc_best.enctype
    
    for up in range(15):
            enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([1,1,0,up+1,up+1]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    with open(figdir+'/QAM_'+ wname[l] +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)