from imports import *
from functions import *
from NN_classes import *


def train_network(M=4,SNR_s = [1,1], SNR_c = [100,200],train_params=[50,100,0.01,1,1],weight_sens = 0.1,max_target=1,stage=None,NNs=None, plotting=True, setbehaviour="none", namespace="",loss_beam=0,enctype="QAM", num_ue=2):
    """ 
    Training process of NN autoencoder

    M : number of constellation symbols
    sigma_n : noise standard deviation 
    train_params=[num_epochs,batches_per_epoch, learn_rate]
    weight_sens : Impact of the radar receiver; impact of communicationn is (1-weight_sens)
    stage : NNs should be trained serially; therefore there are 3 training stages:
        stage = 1: training of encoder, decoder, beamformer and angle estimation
        stage = 2: training of encoder, decoder, beamformer and estimation of angle uncertainty
        stage = 3: training of encoder, decoder, beamformer and target detection 
    
    M, sigma_n, modradius are lists of the same size 
    setbehaviour (str) : encompasses methods to take care of permutations;
        "setloss" -> use hausdorff distance for loss; permute in validation set
        "permute"
        "sortall"
        "sortphi"
        "none" : do nothing
        "ESPRIT" -> use esprit algorithm instead of NN 
    plotting (bool): toggles Plotting of PDFs into folder /figures 
    """
    encoding = ['sum of onehot', 'onehot'][0] #onehot only for a single target implemented for now
    #enctype = "QAM" or "NN"
    benchmark = 1
    canc = 0
    k_a = [16,1]

    torch.set_default_device(device)
    #torch.autograd.set_detect_anomaly(True)
    #if M.size()!=1:
    #    raise error("M, sigma_n, need to be of same size (float)!")
    
    num_epochs=int(train_params[0])
    batches_per_epoch=int(train_params[1]) #np.ceil(2000/num_epochs)on Validation Dataset
    cpr_min = int(train_params[3]) # communication per radar: integer of how many communication symbols are transmitted for the same radar estimation
    cpr_max = int(train_params[4])
    learn_rate =train_params[2]
    N_valid = 10000

    logging.info("Running training in training_routine_multitarget_nofft.py")
    logging.info("Maximum target number is %i" % max_target)
    logging.info("Set behaviour is %s" % setbehaviour )
    logging.info("loss_beam is %s" % str(loss_beam))
    lambda_txr = 0.1

    sigma_nc_all=1/torch.sqrt(SNR_c).to(device)
    sigma_ns_all=1/torch.sqrt(SNR_s).to(device)
    sigma_c=1
    sigma_s=1


    printing=False #suppresses all printed output but GMI
    
    # Generate Validation Data
    #y_valid = torch.zeros(N_valid,dtype=int, device=device).to(device)
    y_valid = torch.randint(0,int(M),(N_valid*cpr_max,num_ue)).to(device)

    if plotting==True:
        # meshgrid for plotting
        ext_max = 2  # assume we normalize the constellation to unit energy than 1.5 should be sufficient in most cases (hopefully)
        mgx,mgy = cp.meshgrid(cp.linspace(-ext_max,ext_max,200), cp.linspace(-ext_max,ext_max,200))
        meshgrid = cp.column_stack((cp.reshape(mgx,(-1,1)),cp.reshape(mgy,(-1,1))))
    
    if NNs==None:
        #enc = QAM_encoder(M)
        if enctype=="NN":
            enc=[]
            for i in range(num_ue):
                enc.append(Encoder(M).to(device))
        else:
            enc=[]
            for i in range(num_ue):
                enc.append(QAM_encoder(M).to(device))
        dec=[]
        for i in range(num_ue):                   
            dec.append(Decoder(M).to(device))
        beam = Beamformer(kx=k_a[0],ky=k_a[1],n_ue=num_ue).to(device)
        rad_rec = Radar_receiver(kx=k_a[0],ky=k_a[1],max_target=max_target,cpr_max=15, encoding=encoding).to(device)
        #rad_rec = Joint_radar_receiver(kx=k_a[0],ky=k_a[1],max_target=max_target, encoding=encoding).to(device)
    else:
        if enctype=="NN":
            enc = NNs[0]
        else:
            enc=[]
            for i in range(num_ue):
                enc.append(QAM_encoder(M).to(device)) 
        dec = NNs[1]
        beam = NNs[2]
        rad_rec = NNs[3]
        encoding = rad_rec.encoding
        
    
    # Adam Optimizer
    # List of optimizers in case we want different learn-rates
    optimizer=[]
    
    for i in range(num_ue):
        if enctype=="NN":
            optimizer.append(optim.Adam(enc[i].parameters(), lr=float(learn_rate)))
        optimizer.append(optim.Adam(dec[i].parameters(), lr=float(learn_rate)))
    optimizer.append(optim.Adam(rad_rec.parameters(), lr=float(learn_rate)))
    optimizer.append(optim.Adam(beam.parameters(), lr=float(learn_rate)))

    softmax = nn.Softmax(dim=1).to(device)

    # Cross Entropy loss
    loss_fn = nn.CrossEntropyLoss()
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")
    bce_loss_target = nn.BCEWithLogitsLoss(reduction="none")
    mse_loss =nn.MSELoss(reduction="none")
    gauss_nll_loss = nn.GaussianNLLLoss()

    # fixed batch size of 10000
    batch_size_per_epoch = np.zeros(num_epochs, dtype=int)+10000*2

    validation_BER = torch.zeros((int(num_epochs),int(torch.log2(M)),num_ue))
    validation_SERs = torch.zeros((int(num_epochs),num_ue))
    validation_received = []
    det_error=[[],[]]
    sent_all=[]
    rmse_angle=[]
    rmse_benchmark=[]
    rmse_benchmark_2=[]
    d_angle_uncertain=[]
    CRB_azimuth = 100 # init high value for CRBs
    CRB_elevation = 100
    print('Start Training stage '+str(stage))
    logging.info("Start Training stage %s" % stage)

    bitnumber = int(torch.sum(torch.log2(M)))
    gmi = torch.zeros(int(num_epochs),device=device)
    gmi_exact = torch.zeros((int(num_epochs), bitnumber,num_ue), device=device)
    #m_enc = torch.zeros((batch_size_per_epoch[epoch]),device=device)
    SNR = np.zeros(int(num_epochs))
    #mradius =[]
    k = torch.arange(beam.kx)+1

    for epoch in range(int(num_epochs)):
        cpr= torch.randint(cpr_min,cpr_max+1,(batch_size_per_epoch[epoch],1),device=device)
        
        #validation_BER.append([])
        system = system_runner(M,SNR_s , SNR_c ,train_params,weight_sens,max_target,NNs, setbehaviour, namespace,enctype, num_ue)
        for step in range(int(batches_per_epoch)):
            decoded, batch_cw, t_NN, target, angle_shrunk, permuted_angle_shrunk, angle_corr, _, _ = system.run_system(train=True)
            
            if stage==1: # Training of angle estimation
                #loss = bceloss_sym(M,(softmax(decoded)), batch_labels.long())
                closs = comm_loss_ra(decoded,batch_cw,alpha=1)#*cpr_max#*torch.unsqueeze((cpr_ex),1))
                angle_loss = torch.mean(mse_loss(torch.squeeze(angle_shrunk), torch.squeeze(permuted_angle_shrunk))*(angle_corr).repeat(2,1).T.reshape(-1,2))
                loss = (1-weight_sens)*closs + weight_sens*angle_loss #+ beamloss
            elif stage==2: # Training of communication
                #loss = bceloss_sym(M,(softmax(decoded)), batch_labels.long())
                loss = torch.mean(torch.sum(bce_loss(decoded,batch_cw),1))#*cpr_max#*torch.unsqueeze((cpr_ex),1))
                loss = (1-weight_sens)*loss.clone() 
            elif stage==3: # Training of target detection
                #loss = bceloss_sym(M,(softmax(decoded)), batch_labels.long())
                closs = comm_loss_ra(decoded,batch_cw,alpha=1)#*cpr_max#*torch.unsqueeze((cpr_ex),1))
                bloss = torch.mean(bce_loss_target(torch.squeeze(t_NN),torch.squeeze(target)))
                loss = (1-weight_sens)*closs + (weight_sens)*(bloss) #+ beamloss # combine with radar detect loss

            else:
                #closs = bceloss_sym(M,(softmax(decoded)), batch_labels.long())
                #closs = torch.mean(torch.sum(bce_loss(decoded[0],batch_cw[0]),1))#*cpr_max#*torch.unsqueeze((cpr_ex),1))
                closs = comm_loss_ra(decoded,batch_cw,alpha=1)
                angle_loss = torch.mean(mse_loss(torch.squeeze(angle_shrunk), torch.squeeze(permuted_angle_shrunk))*(angle_corr).repeat(2,1).T.reshape(-1,2))
                loss =  (1-weight_sens)*closs + (weight_sens)*(torch.mean(bce_loss_target(torch.squeeze(t_NN),torch.squeeze(target)))+angle_loss)# combine with radar detect loss
                #loss = loss.clone() + (weight_sens)*angle_loss ##+angle_loss_spec) #+ torch.mean(torch.abs(modulated)**2) # add angle loss; add signal energy to loss
                
            
            # compute gradients
            loss.backward() 

            # run optimizer
            for elem in optimizer:
                elem.step()
            

            # reset gradients
            for elem in optimizer:
                elem.zero_grad()

        
        with torch.no_grad(): #no gradient required on validation data
            decoded, batch_cw, t_NN, target, angle_shrunk, permuted_angle_shrunk, _, benchmark_angle_nn, sent_all = system.run_system(train=False,cpr=[cpr_max,cpr_max],SNR_c=1/sigma_nc_all[1].repeat(2),SNR_s=1/sigma_ns_all[0].repeat(2))
            y_valid = system.valid_labels

            if plotting==True:
                cvalid = torch.zeros(N_valid*cpr_max)
            #decoded_valid=torch.zeros((N_valid*cpr_max,int(torch.max(M)),num_ue), dtype=torch.float32, device=device)
            
            decoded_hard=[]
            decoded_symbol=[]
            #bits=[]
            s=[]
            for i in range(num_ue):
                gmi_exact[epoch,:,i]=GMI(M,decoded[i], batch_cw[i])
                decoded_hard.append(torch.round(torch.sigmoid(decoded[i])).type(torch.int16).to(device))
                decoded_symbol_h= torch.zeros((N_valid*cpr_max),dtype=torch.long).to(device)
                code_ints = torch.zeros(M).to(device)
                for b in range(int(torch.log2(M))):
                    decoded_symbol_h += decoded_hard[i][:,b]*2**b
                    code_ints += gray_code(M).to(device)[:,b]*2**b
                decoded_symbol.append(code_ints[decoded_symbol_h.detach().cpu()])
                #bits.append(BER(decoded_hard[i], batch_cw[i],M))
                validation_SERs[epoch,i] = SER(decoded_symbol[i], y_valid[:,i])
                validation_BER[epoch,:,i]= BER(decoded_hard[i], batch_cw[i],M)

            # detection probability
            prob_e_d = torch.sum(torch.round(torch.squeeze(t_NN))*(torch.squeeze(target)))/torch.sum(torch.squeeze(target)).to(device)
            # false alarm rate
            prob_f_d = torch.sum(torch.round(torch.squeeze(t_NN))*(1-torch.squeeze(target)))/torch.sum(1-torch.squeeze(target)).to(device)
         
                
            if printing==True:
                print('Detect Probability of radar detection after epoch %d: %f' % (epoch, prob_e_d))            
                print('False Alarm rate of radar detection after epoch %d: %f' % (epoch, prob_f_d))
            logging.info('Detect Probability of radar detection after epoch %d: %f' % (epoch, prob_e_d))
            logging.info('False Alarm rate of radar detection after epoch %d: %f' % (epoch, prob_f_d))
            #if l_target==max_target-1:
            det_error[0].append(prob_e_d.detach().cpu().numpy())
            det_error[1].append(prob_f_d.detach().cpu().numpy())

            ## Angle estimation
            x_detect = torch.nonzero(t_NN*target > 0.5).to(device)[:,0] # targets that were present and were detected
            rmse_benchmark.append(torch.sqrt(torch.mean(torch.abs((benchmark_angle_nn[x_detect,0,:] - permuted_angle_shrunk[x_detect,0,:]))**2)).detach().cpu().numpy())
            rmse_angle.append(torch.sqrt(torch.mean(torch.abs(torch.squeeze(angle_shrunk[x_detect,0,:])-torch.squeeze(permuted_angle_shrunk[x_detect,0,:]))**2)).detach().cpu().numpy())
  
            if printing==True:
                print('Angle estimation error after epoch %d: %f (rad)' % (epoch, rmse_angle[epoch]))
            logging.info('Angle estimation error after epoch %d: %f (deg) | %f (rad)' % (epoch, 180/np.pi*rmse_angle[epoch],rmse_angle[epoch]))

            # color map for plot
            if plotting==True:
                cvalid=y_valid
            

            if printing==True:
                print('Validation BER after epoch %d: ' % (epoch) + str(validation_BER[epoch]) +' (loss %1.8f)' % (loss.detach().cpu().numpy()))  
                print('Validation SER after epoch %d: %f (loss %1.8f)' % (epoch, validation_SERs[epoch], loss.detach().cpu().numpy()))              
            
            logging.debug('Validation BER after epoch %d: ' % (epoch) + str(validation_BER[epoch]) +' (loss %1.8f)' % (loss.detach().cpu().numpy()))
            logging.debug('Validation SER after epoch %d: ' % (epoch) + str(validation_SERs[epoch]) +' (loss %1.8f)' % (loss.detach().cpu().numpy()))
        
            if printing==True:
                print("GMI is: "+ str(torch.sum(gmi_exact[epoch]).item()) + " bit after epoch %d (loss: %1.8f)" %(epoch,loss.detach().cpu().numpy()))
            logging.info("GMI is: "+ str(torch.sum(gmi_exact[epoch]).item()) + " bit after epoch %d (loss: %1.8f)" %(epoch,loss.detach().cpu().numpy()))

            # Choose best training epoch to save NNs or keep training further
            loss_ev = (-torch.sum(gmi_exact[epoch])-prob_e_d+prob_f_d).detach().cpu().numpy() + rmse_angle[epoch] #+ torch.mean((torch.log10(uncertainty+1e-9)+(torch.abs(angle*np.pi - phi_valid)**2)/(2*uncertainty**2+1e-9))*t)


            if epoch==0:
                enc_best=system.enc
                dec_best=system.dec
                best_epoch=0
                beam_best=system.beam
                rad_rec_best = system.rad_rec
                loss_b = 100
            elif epoch==num_epochs-1:
                enc_best=system.enc
                dec_best=system.dec
                best_epoch=epoch
                beam_best=system.beam
                rad_rec_best = system.rad_rec
                loss_b = loss_ev
            elif loss_ev<loss_b:
                enc_best=system.enc
                dec_best=system.dec
                best_epoch=epoch
                beam_best=system.beam
                rad_rec_best = system.rad_rec
                loss_b = loss_ev

            # mmse = torch.conj(CSI)/(torch.conj(CSI)*CSI+torch.squeeze(sigma_nc**2)).to(device)
            # validation_received.append((channel*mmse).detach().cpu().numpy())
            # sent_all.append(modulated.detach().cpu().numpy())

            # sig = torch.sum(direction[:,:,0] * radiate(beam.kx,torch.tensor([0],device=device), beam.ky, torch.tensor([np.pi/2],device=device)), axis=1)**2 
            # SNR = np.abs((sig*sigma_s**2/sigma_ns**2).detach().cpu().numpy())
            # CRB_azimuthi = 6*4/((2*np.pi)**2*np.cos(0*np.pi/180)**2*SNR*rad_rec.k[0].detach().cpu().numpy()*(rad_rec.k[0].detach().cpu().numpy()**2-1+1e-6))/(rad_rec.k[1].detach().cpu().numpy()*torch.mean(cpr.float()).detach().cpu().numpy())
            # CRB_elevationi = 6*4/((2*np.pi)**2*np.cos(70*np.pi/180)**2*SNR*rad_rec.k[1].detach().cpu().numpy()*(rad_rec.k[1].detach().cpu().numpy()**2-1+1e-6))/(rad_rec.k[0].detach().cpu().numpy()*torch.mean(cpr.float()).detach().cpu().numpy())

            # CRB_azimuth = np.minimum(CRB_azimuth,CRB_azimuthi)
            # CRB_elevation = np.minimum(CRB_elevation,CRB_elevationi)

    if enctype=="NN":
        constellations = cp.zeros((M,num_ue),dtype=np.complex128)
        for i in range(num_ue):
            constellations[:,i] = cp.asarray(enc_best[i](torch.eye(int(M), device=device), noise=torch.unsqueeze(sigma_nc_all[1],0).repeat(M,1)).cpu().detach().numpy())
    else:
        constellations = cp.zeros((M,num_ue),dtype=np.complex128)
        for i in range(num_ue):
            constellations[:,i] = enc[i](torch.arange(int(M))).cpu().detach().numpy()
    logging.info("Constellation is: %s" % (str(constellations)))

    ### Plot & collect the results in log:
    # if plotting==True:
    #     decision_region_evolution = []
    #     for i in range(len(dec_best)):
    #         mesh_bits = torch.round((torch.sigmoid(dec_best[i]((torch.view_as_complex(torch.Tensor(meshgrid))).to(device)*CSI[0],CSI[0].repeat(len(meshgrid)),torch.unsqueeze(sigma_nc,0).repeat(len(meshgrid),1))))).type(torch.int16).to(device) #*torch.exp(-1j*torch.angle(CSI_best[0]))
    #         mesh_prediction= torch.zeros((len(meshgrid)),dtype=torch.long).to(device)
    #         for b in range(int(torch.log2(M))):
    #             mesh_prediction += mesh_bits[:,b]*2**b
    #         decision_region_evolution.append(0.195*mesh_prediction.detach().cpu().numpy() + 0.4)

    print('Training finished')
    logging.info('Training finished')

    logging.info("SER obtained: %s" % (str(validation_SERs)))
    logging.info("GMI obtained: %s" % str(np.sum(gmi_exact.detach().cpu().numpy(),axis=1)))
      

    logging.info("CRB in azimuth is: "+str(CRB_azimuth))
    logging.info("CRB in elevation is: "+str(CRB_elevation))

    beam_tensor = np.pi/180*torch.tensor([[-20.0,20.0,50,70]])
    direction = system.beam(beam_tensor)
    
    if plotting==True:
        if device=='cpu':
            plot_training(validation_SERs.cpu().detach().numpy(),validation_BER,M, constellations, gmi_exact.detach().cpu().numpy(), direction, det_error, np.asarray(rmse_angle),benchmark = rmse_benchmark, stage=stage,namespace=namespace, CRB=CRB_azimuth, antennas=rad_rec.k,enctype=enctype) 
        else:
            try:
                plot_training(validation_SERs.cpu().detach().numpy(), validation_BER,M, cp.asnumpy(constellations),  gmi_exact.detach().cpu().numpy(), direction, det_error, np.asarray(rmse_angle),benchmark = rmse_benchmark, stage=stage, namespace=namespace, CRB=CRB_azimuth,antennas=rad_rec.k,enctype=enctype) 
            except:
                pass
                #plot_training(validation_SERs.cpu().detach().numpy(), validation_received[best_epoch],cvalid,M, constellations, gmi, decision_region_evolution, meshgrid, gmi_exact.detach().cpu().numpy(), sent_all, det_error, np.asarray(rmse_angle),benchmark = rmse_benchmark, stage=stage, namespace=namespace, CRB=CRB_azimuth) 
            plot_training(validation_SERs.cpu().detach().numpy(), validation_BER,M, constellations,  gmi_exact.detach().cpu().numpy(), direction, det_error, np.asarray(rmse_angle),benchmark = rmse_benchmark, stage=stage, namespace=namespace, CRB=CRB_azimuth, antennas=rad_rec.k,enctype=enctype) 

            #path = figdir+"/plots"+".pkl"
            #with open(path, 'wb') as fh:
                #pickle.dump([validation_SERs.cpu().detach().numpy(), np.array(validation_received[best_epoch]),cvalid,M, constellations, gmi, decision_region_evolution, meshgrid, gmi_exact.detach().cpu().numpy(), sent_all, det_error, np.asarray(rmse_angle)], fh)
    if device=='cpu':
        return(enc_best,dec_best, beam_best, rad_rec_best, validation_SERs,gmi_exact, det_error, cp.array(constellations))
    else:
        try:
            return(enc_best,dec_best, beam_best, rad_rec_best, validation_SERs,gmi_exact, det_error, cp.asnumpy(constellations))
        except:
            return(enc_best,dec_best, beam_best, rad_rec_best, validation_SERs,gmi_exact, det_error, (constellations))







