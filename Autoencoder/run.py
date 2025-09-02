import datetime
import pickle
import sys
import imports



begin_time = datetime.datetime.now()
#### Enable setting arguments from command line
if len(sys.argv)==1:
    choice = 18
elif len(sys.argv)==2:
    choice = int(sys.argv[1])
elif len(sys.argv)==4:
    choice=0
    from training_routine_SNRsweep import *
    logging.info("One simulation with SNR sweep and 1 targ, NN or QAM enc")
    enctype = str(sys.argv[1])
    M=torch.tensor([int(sys.argv[2])], dtype=int).to(device)
    wr = float(sys.argv[3])
    #sigma_n=torch.tensor([0.1], dtype=float, device=device)
    #sigma_c=100*sigma_n
    #sigma_s=torch.tensor([0.1]).to(device)
    SNR_s = torch.pow(10.0,torch.tensor([-10,10],device=device)/10)
    SNR_c = torch.pow(10.0,torch.tensor([5,30],device=device)/10)
    #training of exact beamform
    logging.info("Modulation Symbols: "+str(M))
    logging.info("SNR sensing = "+str(SNR_s))
    logging.info("SNR Communication = "+str(SNR_c))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=3, plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=1,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    #setting levels
    wrstr = str(wr).replace(".","")
    while wrstr[-1]=="0":
        wrstr = wrstr[0:len(wrstr)-1]
    for up in range(15):
        enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([1,1,0,up+1,up+1]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    with open(figdir+'/'+enctype+'_'+ wrstr +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif len(sys.argv)==5:
    choice=0
    from training_routine_SNRsweep import *
    logging.info("One simulation with SNR sweep and 1 targ, NN enc")
    enctype = str(sys.argv[1])
    M=torch.tensor([int(sys.argv[2])], dtype=int).to(device)
    wr = float(sys.argv[3])
    #sigma_n=torch.tensor([0.1], dtype=float, device=device)
    #sigma_c=100*sigma_n
    #sigma_s=torch.tensor([0.1]).to(device)
    SNR_s = torch.pow(10.0,torch.tensor([-10,10],device=device)/10)
    SNR_c = torch.pow(10.0,torch.tensor([5,30],device=device)/10)
    #training of exact beamform
    logging.info("Modulation Symbols: "+str(M))
    logging.info("SNR sensing = "+str(SNR_s))
    logging.info("SNR Communication = "+str(SNR_c))
    if device!='cpu':
        if enctype=="QAM":
            enc, dec, beam, rad_rec = pickle.load( open( 'set/final_1508/QAM/QAM_07.pkl', "rb" ) )
        else:
            enc, dec, beam, rad_rec = pickle.load( open( 'set/final_1508/NN/NN_07.pkl', "rb" ) )
        enc.to(device)
        dec.to(device)
        beam.to(device)
        rad_rec.to(device)
    else:
        if enctype=="QAM":
            enc, dec, beam, rad_rec = CPU_Unpickler( open( 'set/final_1508/QAM/QAM_07.pkl', "rb")).load()
        else:
            enc, dec, beam, rad_rec = CPU_Unpickler( open( 'set/final_1508/NN/NN_07.pkl', "rb")).load()
    #enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=3, plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    #enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=1,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    #enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([100,50,0.0001,1,15]),weight_sens=wr,max_target=1,stage=None,NNs=[enc,dec, beam, rad_rec], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    wrstr = str(wr).replace(".","")
    while wrstr[-1]=="0":
        wrstr = wrstr[0:len(wrstr)-1]
    with open(figdir+'/'+enctype+'_temp_'+ wrstr +'.pkl', 'wb') as fh: # saving intermediate results
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
    #setting levels
    for up in range(15):
        enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([1,1,0,up+1,up+1]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=False,setbehaviour="none", namespace=namespace, enctype=enctype)
    with open(figdir+'/'+enctype+'_'+ wrstr +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
else:
    print(len(sys.argv))

""" ## choice controls the kind of simulation ##
choice = 11    : 1 target and constellation shaping
choice = 17    : 1 target and QAM modulation
choice = 18    : 1 target and 2 UEs with QAM modulation
"""

if choice == 0:
    pass
elif choice == 11:
    from training_routine_SNRsweep import *
    logging.info("One simulation with 1 Targets with constellation shaping")
    M=torch.tensor([16], dtype=int).to(device)
    #sigma_n=torch.tensor([0.1], dtype=float, device=device)
    #sigma_c=100*sigma_n
    #sigma_s=torch.tensor([0.1]).to(device)
    SNR_s = torch.tensor([0.1,10],device=device)
    SNR_c = torch.tensor([10,100],device=device)
    #training of exact beamform
    logging.info("Detection and Localization of 1 target")
    logging.info("Modulation Symbols: "+str(M))
    logging.info("SNR sensing = "+str(SNR_s))
    logging.info("SNR Communication = "+str(SNR_c))
    #logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1,1]),weight_sens=0.9,max_target=1,stage=None, plotting=True,setbehaviour="none", namespace=namespace, enctype="NN")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1,3]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="NN")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1,8]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="NN")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1,10]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="NN")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.001,1,12]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="NN")
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif choice == 17:
    from training_routine_SNRsweep import *
    logging.info("One simulation with SNR sweep and 1 targ, QAM enc")
    M=torch.tensor([16], dtype=int).to(device)
    #sigma_n=torch.tensor([0.1], dtype=float, device=device)
    #sigma_c=100*sigma_n
    #sigma_s=torch.tensor([0.1]).to(device)
    SNR_s = torch.pow(10.0,torch.tensor([-1,1],device=device))
    SNR_c = torch.pow(10.0,torch.tensor([0,2],device=device))
    #training of exact beamform
    logging.info("Modulation Symbols: "+str(M))
    logging.info("SNR sensing = "+str(SNR_s))
    logging.info("SNR Communication = "+str(SNR_c))
    #logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,1]),weight_sens=0.9,max_target=1,stage=3, plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,1]),weight_sens=0.9,max_target=1,stage=1,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,1]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,3]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,5]),weight_sens=0.9,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM")
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
elif choice == 18:
    from training_routine_SNRsweep import *
    logging.info("One simulation with SNR sweep and 1 targ, NN enc, 2 UEs")
    M=torch.tensor([16], dtype=int).to(device)
    #sigma_n=torch.tensor([0.1], dtype=float, device=device)
    #sigma_c=100*sigma_n
    #sigma_s=torch.tensor([0.1]).to(device)
    SNR_s = torch.pow(10.0,torch.tensor([-10,10],device=device)/10)
    SNR_c = torch.pow(10.0,torch.tensor([10,30],device=device)/10)
    #training of exact beamform
    logging.info("Modulation Symbols: "+str(M))
    logging.info("SNR sensing = "+str(SNR_s))
    logging.info("SNR Communication = "+str(SNR_c))
    #logging.info("Radar channel, Swerling1 with sigma_s = "+str(sigma_s))
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,1]),weight_sens=0.7,max_target=1,stage=3, plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=2)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,1]),weight_sens=0.7,max_target=1,stage=1,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=2)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,1]),weight_sens=0.7,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=2)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,3]),weight_sens=0.7,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=2)
    enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.001,1,15]),weight_sens=0.7,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=2)
    with open(figdir+'/trained_NNs_'+ namespace +'.pkl', 'wb') as fh:
        pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)
else:
    pass
logging.info("Training duration is" + str(datetime.datetime.now()-begin_time))