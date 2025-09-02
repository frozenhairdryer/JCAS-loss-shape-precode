from numpy import empty
from imports import *
from functions import *

class system_runner():
    def __init__(self,M=4,SNR_s = [1,1], SNR_c = [100,200],train_params=[50,100,0.01,1,1],weight_sens = 0.1,max_target=1,NNs=None, setbehaviour="none", namespace="",enctype="QAM", num_ue=2, N_valid=10000):
        self.enctype = enctype
        self.M = M
        self.SNR_s = SNR_s
        self.SNR_c = SNR_c
        self.train_params=train_params
        self.weight_sens = weight_sens
        self.max_target = max_target
        self.setbehaviour=setbehaviour
        self.namespace= namespace
        self.num_ue = num_ue

        k_a = [16,1] # Antenna Array 

        if NNs==None:
            #enc = QAM_encoder(M)
            if enctype=="NN":
                self.enc=[]
                for i in range(num_ue):
                    self.enc.append(Encoder(M).to(device))
            else:
                self.enc=[]
                for i in range(num_ue):
                    self.enc.append(QAM_encoder(M).to(device))
            self.dec=[]
            for i in range(num_ue):                   
                self.dec.append(Decoder(M).to(device))
            self.beam = Beamformer(kx=k_a[0],ky=k_a[1],n_ue=num_ue).to(device)
            self.rad_rec = Radar_receiver(kx=k_a[0],ky=k_a[1],max_target=max_target,cpr_max=15).to(device)
            #rad_rec = Joint_radar_receiver(kx=k_a[0],ky=k_a[1],max_target=max_target, encoding=encoding).to(device)
        else:
            if enctype=="NN":
                self.enc = NNs[0]
            else:
                self.enc=[]
                for i in range(num_ue):
                    self.enc.append(QAM_encoder(M).to(device)) 
            self.dec = NNs[1]
            self.beam = NNs[2]
            self.rad_rec = NNs[3]
        
        self.N_valid = N_valid
        batch_labels = torch.zeros((self.N_valid*int(self.train_params[4])*self.num_ue),dtype=torch.long, device=device)
        batch_labels.random_(int(self.M)).to(device)
        
        self.labels_onehot = torch.zeros(int(self.N_valid*int(self.train_params[4])*self.num_ue), int(self.M), device=device)
        self.labels_onehot[range(self.labels_onehot.shape[0]), batch_labels.long()]=1
        self.labels_onehot = self.labels_onehot.reshape(int(self.N_valid*int(self.train_params[4])), int(self.M),self.num_ue)
        batch_labels = batch_labels.reshape(-1,self.num_ue)
        self.valid_labels=batch_labels
        



    def run_system(self,train = True, cpr=None, SNR_s=None, SNR_c=None):
        if train==True:
            N_valid=10000*2 #np.ceil(2000/num_epochs)on Validation Dataset
            cpr_min = int(self.train_params[3]) # communication per radar: integer of how many communication symbols are transmitted for the same radar estimation
            cpr_max = int(self.train_params[4])

            if self.enctype=="NN":
                for i in range(self.num_ue):
                    self.enc[i].train()
            for i in range(self.num_ue):
                self.dec[i].train()
            self.rad_rec.train()
            self.beam.train()
            
            sigma_nc_all=1/torch.sqrt(self.SNR_c).to(device)
            sigma_ns_all=1/torch.sqrt(self.SNR_s).to(device)
            
        else:
            N_valid = self.N_valid
            #N_valid = int(self.train_params[1])
            self.rad_rec.eval()
            self.beam.eval()
            if self.enctype=="NN":
                for i in range(self.num_ue):
                    self.enc[i].eval()
            cpr_min=cpr[0]
            cpr_max=cpr[1]
            sigma_nc_all=1/torch.sqrt(SNR_c).to(device)
            sigma_ns_all=1/torch.sqrt(SNR_s).to(device)
            
        
        #### random parameters
        
        cpr= torch.randint(cpr_min,cpr_max+1,(N_valid,1),device=device)
        
        
        sigma_nc = torch.rand((int(N_valid*cpr_max),self.num_ue)).to(device)*(sigma_nc_all[1]-sigma_nc_all[0])+sigma_nc_all[0]
        sigma_ns = (torch.rand(int(N_valid),1).to(device)*(sigma_ns_all[1]-sigma_ns_all[0])+sigma_ns_all[0])
        sigma_ns_ex = sigma_ns.repeat(1,cpr_max).reshape(int(N_valid)*cpr_max)
        # Generate training data: In most cases, you have a dataset and do not generate a training dataset during training loop
        # sample new mini-batch directory on the GPU (if available)
        decoded=torch.zeros(int(N_valid*cpr_max),(torch.max(self.M)), device=device)
        
        ### Initialize random communication symbols, receiver angles, target numbers and target angles ###
        if train==True:
            batch_labels = torch.zeros((N_valid*cpr_max*self.num_ue),dtype=torch.long, device=device)
            batch_labels.random_(int(self.M)).to(device)
            #batch_labels[:10] = torch.tensor([0,1,2,3,0,1,2,3,0,1])
            batch_labels_onehot = torch.zeros(int(N_valid*cpr_max*self.num_ue), int(self.M), device=device)
            batch_labels_onehot[range(batch_labels_onehot.shape[0]), batch_labels.long()]=1
            batch_labels_onehot = batch_labels_onehot.reshape(int(N_valid*cpr_max), int(self.M),self.num_ue)
            batch_labels = batch_labels.reshape(-1,self.num_ue)
        else:
            batch_labels = self.valid_labels
            batch_labels_onehot = self.labels_onehot
        
        batch_cw = [] #torch.zeros(int(N_valid*cpr_max), int(np.log2(M)),num_ue)
        if self.enctype!="QAM":
            for i in range(self.num_ue):
                batch_cw.append(gray_code(self.M)[batch_labels[:,i]].to(device))
        else:
            for i in range(self.num_ue):
                batch_cw.append(self.enc[i].coding()[batch_labels[:,i]])

        theta_valid = torch.zeros((int(N_valid),2,self.num_ue), device=device)
        theta_valid[:,0,:] = np.pi/180*torch.tensor([50,70]).to(device)#(torch.rand(int(N_valid),num_ue)*20+30)
        theta_valid[:,1,:] = np.pi/2#/180*(torch.rand((int(N_valid)))*1+90) # between 90 and 100 deg
        theta_valid = torch.repeat_interleave(theta_valid,cpr_max,dim=0)
        
        target_labels = torch.randint(self.max_target+1,(int(N_valid),)).to(device) # Train in each epoch first 1 target, then 2, then ...

        ##encoding: [1,1,0,...] means 2 targets are detected
        target = torch.zeros((int(N_valid),self.max_target)).to(device)
        label_tensor = torch.zeros(self.max_target+1,self.max_target).to(device)
        for x in range(self.max_target+1):
            label_tensor[x] = torch.concat((torch.ones(x), torch.zeros(self.max_target-x)))
        target += label_tensor[target_labels] 
        target_onehot = torch.zeros((int(N_valid),self.max_target+1)).to(device)
        target_onehot[np.arange(int(N_valid)),target_labels] = 1


        phi_valid = torch.zeros((int(N_valid),2,self.max_target)).to(device)
        phi_valid[:,0,:] = np.pi/180*(torch.rand((int(N_valid),self.max_target))*40-20) # paper: [-20 deg, 20 deg]
        phi_valid[:,1,:] = np.pi/2#/180*(torch.rand((int(N_valid),max_target))*10+90) # between 90 and 100 deg

        phi_valid[:,0,:] *= target
        phi_valid[:,1,:] *= target

        # Only applies for multiple targets
        if (self.setbehaviour=="sortall" or self.setbehaviour=="sortphi") and self.max_target>1:
            for l in range(int(N_valid)):
                idx1 = torch.nonzero(phi_valid[l,0,:]).to(device)
                idx0 = torch.nonzero(phi_valid[l,0,:] == 0).to(device)
                if idx1.numel() > 0:
                    idx = torch.argsort(phi_valid[l,0,torch.squeeze(idx1,1)], descending=True).to(device)
                    if idx0.numel() > 0:
                        idx = torch.squeeze(torch.cat((idx,torch.squeeze(idx0,1))))
                    else:
                        idx = torch.squeeze(idx)
                else:
                    idx = torch.squeeze(idx0).to(device)
                phi_valid[l,0,:] = phi_valid[l,0,idx]
                phi_valid[l,1,:] = phi_valid[l,1,idx]

        # enable oversampling
        phi_valid_ex = phi_valid.repeat(1,cpr_max,1).reshape(cpr_max*N_valid,2,self.max_target)

        input_beam = torch.zeros((int(N_valid*cpr_max),4)).to(device)
        input_beam[:,0:2] = np.pi/180*torch.tensor([[-20.0,20.0]]).repeat(int(N_valid*cpr_max),1).to(device)
        input_beam[:,2:4] = theta_valid[:,0,0:2]
        direction = self.beam(input_beam).to(device) # give the angles in which targets/receivers are to be expected
        
        if direction.isnan().any():
            raise RuntimeError("NaN encountered while training.")

        # Propagate (training) data through transmitter
        if self.enctype!="QAM":
            encoded = torch.zeros((N_valid*cpr_max,self.num_ue,1)).type(torch.complex64)
            for i in range(self.num_ue):
                encoded[:,i] = torch.unsqueeze(self.enc[i](batch_labels_onehot[:,:,i],noise=sigma_nc[:,i]),1)
            del batch_labels_onehot
        else:
            encoded = torch.zeros((N_valid*cpr_max,self.num_ue,1)).type(torch.complex64)
            for i in range(self.num_ue):
                encoded[:,i] = torch.unsqueeze(self.enc[i](batch_labels[:,i]),1).to(device)
        

        #modulated = torch.sum(encoded.expand(N_valid*cpr_max,num_ue,k_a[0]*k_a[1]).mT @ direction.mT,1)
        #modulated = torch.matmul(encoded, torch.unsqueeze(torch.transpose(direction,0,1),0)) # Apply Beamforming 
        modulated = torch.squeeze(direction @ encoded).to(device)

        # Propagate through channel
        received = [] #torch.zeros(N_valid*cpr_max,num_ue).type(torch.complex64)
        decoded = [] #torch.zeros(N_valid*cpr_max,int(np.log2(M)),num_ue)
        for i in range(self.num_ue):
            to_receiver = torch.sum(modulated * radiate(self.beam.kx,theta_valid[:,0,i], self.beam.ky, theta_valid[:,1,i]), axis=1)
            r, beta = rayleigh_channel(to_receiver, 1, sigma_nc[:,i], 0.1)
            received.append(r)
            # calculate specific channel state information
            CSI = beta * torch.sum(torch.sum(direction * torch.unsqueeze(radiate(self.beam.kx,theta_valid[:,0,i], self.beam.ky, theta_valid[:,1,i]),2), axis=1),axis=1) 
            decoded.append(self.dec[i](received[i], CSI,noise=sigma_nc[:,i]))

        # radar target detection
        target_ex = target.repeat(1,1,cpr_max).reshape(cpr_max*N_valid,self.max_target)

        received_rad,_ = radar_channel_swerling1(modulated,1, sigma_ns_ex, 0.1,self.rad_rec.k, phi_valid=phi_valid_ex, target=target_ex)
        
        cpr_tensor = torch.tril(torch.ones(cpr_max,cpr_max))
        x_j = torch.transpose(cpr_tensor[cpr.detach().cpu().numpy()-1],0,1).reshape((cpr_max*N_valid,1)).repeat((1,self.rad_rec.k[0])).to(device)
        x_i = torch.reshape(received_rad * x_j, (N_valid, cpr_max, self.rad_rec.k[0], self.rad_rec.k[1])).to(device)
        x_i = torch.transpose(x_i, 2,3)
        x_i = torch.transpose(x_i, 2,1)
        x_i = torch.transpose(torch.reshape(x_i,(N_valid,cpr_max*self.rad_rec.k[1],self.rad_rec.k[0])),1,2)
        R = (x_i @ torch.transpose(torch.conj(x_i),1,2))/cpr.reshape((-1,1,1)).repeat(1,self.rad_rec.k[0],self.rad_rec.k[0])
        #R = R/torch.sum(torch.abs(R)**2) # normalization
        #### Makes everything crash at the 
        #received_rad_phase = received_rad  * cpr_tensor[cpr-1] * torch.exp(-1j*torch.angle(received_rad[:,0])).reshape((-1,1)).repeat(1,rad_rec.k[0])
        #received_avg = torch.sum(torch.reshape(received_rad_phase, (N_valid, cpr_max, rad_rec.k[0], rad_rec.k[1])), dim=1)/cpr
        received_radnn = (R.reshape(N_valid,self.rad_rec.k[0]**2)).to(device)
        t_NN, angle = self.rad_rec(received_radnn, target,cpr=cpr,noise=sigma_ns)#torch.tensor([1], device=device, dtype=torch.float32))
        del x_i
        del x_j

        if self.setbehaviour=="ESPRIT":
            for i in range(int(N_valid)):
                if target_labels[i]!=0:
                    angle[i,0,0:target_labels[i]] = esprit_angle_nns(received_rad[i*cpr:(i+1)*cpr,:],self.rad_rec.k,target_labels[i], cpr, 1)
                    angle[i,1,0:target_labels[i]] = esprit_angle_nns(received_rad[i*cpr:(i+1)*cpr,:],self.rad_rec.k[[1,0]],target_labels[i], cpr, 1)+np.pi/2
        angle[:,0,:] = target * angle.clone()[:,0,:]
        angle[:,1,:] = target * angle.clone()[:,1,:] 
        t_NN = torch.squeeze(t_NN) # Leave as LLRs -> BCEwithlogitsloss doesn't need sigmoid; works with LLRs
        
        t_NN = torch.mean(t_NN.reshape(N_valid,1,self.max_target),1) # Mean of LLRs
        

        # only relevant for multiple targets
        if (self.setbehaviour=="sortall" and self.max_target>1):
            for l in range(int(N_valid*1)):
                t1 = torch.nonzero(torch.abs(angle[l,1,:])>0.1).to(device)
                t0 = torch.nonzero(torch.abs(angle[l,1,:]) <= 0.1).to(device)
                if t1.numel() > 0:
                    idx = torch.argsort(angle[l,0,torch.squeeze(t1,1)], descending=True).to(device)
                    if t0.numel() > 0:
                        idx = torch.squeeze(torch.cat((idx,torch.squeeze(t0,1))))
                    else:
                        idx = torch.squeeze(idx)
                else:
                    idx = torch.squeeze(t0).to(device)
                angle[l,0,:] = angle[l,0,idx]
                angle[l,1,:] = angle[l,1,idx]

        
        permuted_angle = phi_valid
        #
        permuted_angle_shrunk = permuted_angle#torch.mean(permuted_angle.reshape(N_valid,cpr_max,2,max_target),1)
        angle_shrunk = torch.mean(angle.reshape(N_valid,1,2,self.max_target),1)
        #targ = torch.squeeze(torch.nonzero(permuted_angle_shrunk[:,0,0]))
        benchmark_angle_nn = torch.zeros(N_valid,2,self.max_target).to(device) 
        if train==False:
            for i in range(N_valid):
                if target_labels[i]!=0:
                    c = received_rad[i*cpr_max:i*cpr_max+cpr[i],:]
                    time_esprit = datetime.datetime.now()
                    benchmark_angle_nn[i,0,0:target_labels[i]] = esprit_angle_nns(c,self.rad_rec.k,target_labels[i], cpr[i],0)
                    time_esprit = datetime.datetime.now() -time_esprit
                    benchmark_angle_nn[i,1,0:target_labels[i]] = esprit_angle_nns(c,self.rad_rec.k[[1,0]],target_labels[i], cpr[i],0)+np.pi/2

        
        return decoded, batch_cw, t_NN, target, angle_shrunk, permuted_angle_shrunk, (cpr/sigma_ns**2), benchmark_angle_nn, modulated 


class Encoder(nn.Module):
    def __init__(self,M):
        super(Encoder, self).__init__()
        self.M = torch.as_tensor(M, device=device)
        self.K = 16
        self.enctype = "NN"
        # Define Transmitter Layer: Linear function, M icput neurons (symbols), 2 output neurons (real and imaginary part)        
        self.fcT1 = nn.Linear(self.M,8*self.M, device=device) 
        #self.fcT1s = nn.Linear(1,2*self.M, device=device) 
        self.fcT2 = nn.Linear(8*self.M, 8*self.M,device=device)
        self.fcT3 = nn.Linear(8*self.M, 8*self.M,device=device) 
        self.fcT5 = nn.Linear(8*self.M, 2,device=device)
        #if mradius==1:
        #    self.modradius = nn.Parameter(mradius.clone().detach(), requires_grad=False).cuda()
        #else:

        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ReLU()  # in paper: LeakyReLU for hidden layers    


    def forward(self, x, noise):
        # compute output
        #AP = torch.zeros(len(x),1)+0.1
        #x_con = torch.cat((x,AP),1)
        out = self.activation_function(self.fcT1(x))
        
        out = self.activation_function(self.fcT2(out))
        out = self.activation_function(self.fcT3(out))
        encoded = self.fcT5(out)
        # compute normalization factor and normalize channel output
        norm_factor = torch.sqrt(torch.mean(torch.abs((torch.view_as_complex(encoded)).flatten())**2)) # normalize mean squared amplitude to 1
        lim = torch.max(torch.tensor([norm_factor,1])) # Allow for energy reduction
        #norm_factor = torch.max(0.1, norm_factor)
        #norm_factor = torch.max(torch.abs(torch.view_as_complex(encoded)).flatten())
        #if norm_factor>1:        
        #norm_factor = torch.sqrt(torch.mean(torch.mul(encoded,encoded)) * 2 ) # normalize mean amplitude in real and imag to sqrt(1/2)
        modulated = torch.view_as_complex(encoded)/norm_factor
        return modulated

class Decoder(nn.Module):
    def __init__(self,M):
        super(Decoder, self).__init__()
        # Define Receiver Layer: Linear function, 2 icput neurons (real and imaginary part), M output neurons (symbols)
        self.M = torch.as_tensor(M, device=device)
        self.fcR1 = nn.Linear(5,10*self.M,device=device)
        self.dropout1 = nn.Dropout()
        #self.fcR1s = nn.Linear(1,2*self.M, device=device)  
        self.fcR2 = nn.Linear(10*self.M,10*self.M,device=device)
        self.fcR3 = nn.Linear(10*self.M,10*self.M,device=device)
        self.dropout2 = nn.Dropout()
        #self.fcR3b = nn.Linear(20*self.M,20*self.M,device=device)
        self.fcR4 = nn.Linear(10*self.M,10*self.M,device=device) 
        self.fcR5 = nn.Linear(10*self.M, int(torch.log2(self.M)),device=device) 
        #self.alpha=torch.tensor([alph,alph])
        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ELU()
        #self.CSI = CSI #channel state information kappa      

    def forward(self, x, CSI,noise):
        # compute output

        # MMSE equalizer
        mmse = torch.conj(CSI)/(torch.conj(CSI)*CSI+torch.squeeze(noise**2)).to(device)
        x_prep = x*mmse # MMSE equalizer approach
        x_real = torch.cat((torch.view_as_real(x_prep).float(),torch.view_as_real(CSI),(noise**2).reshape(-1,1)),1)#noise**2
        out = self.activation_function(self.fcR1(x_real))
        out = self.activation_function(self.fcR2(out))
        out = self.activation_function(self.fcR3(out))
        out = self.activation_function(self.fcR4(out)) 
        logits = self.fcR5(out)
        
        return logits

class Beamformer(nn.Module):
    """ Transforms the single transmit signal into a tensor of transmit signals for each Antenna
    beamforming is necessary if multiple receivers are present or additional Radar detection is implemented.
    Learning of an appropriate beamforming:
    
    input paramters:
        [theta_min, theta_max] : Angle Interval of Radar target
        [phi_min, phi_max] : Angle interval of receiver
        theta_last : last detected angle of target

    output parameters:
        out : phase shift for transmit antennas
     """
    def __init__(self,kx,ky=1,n_ue=1):
        super(Beamformer, self).__init__()
        self.kx =torch.as_tensor(kx) # Number of transmit antennas in x-direction;
        self.ky =torch.as_tensor(ky) # Number of transmit antennas in y-direction;
        self.n_ue = n_ue
        self.fcB1 = nn.Linear(4,self.kx, device=device)
        self.fcB2 = nn.Linear(self.kx,self.kx, device=device)
        self.fcB3 = nn.Linear(self.kx,self.kx*2, device=device)
        self.fcB4 = nn.Linear(self.kx*2,self.kx*2*n_ue, device=device) # linear output layer

        if ky>1:
            self.fcA1 = nn.Linear(4,self.ky, device=device)
            self.fcA2 = nn.Linear(self.ky,self.ky, device=device)
            self.fcA3 = nn.Linear(self.ky,self.ky*2, device=device)
            self.fcA4 = nn.Linear(self.ky*2,self.ky*2*n_ue, device=device) # linear output layer
        # Non-linearity (used in transmitter and receiver) ?
        self.activation_function = nn.ELU()
    
    def forward(self, Theta):
        try:
            out = self.activation_function(self.fcB1(Theta)).to(device)
        except:
            out = self.activation_function(self.fcB1(torch.hstack((Theta,torch.zeros((len(Theta),1),device=device))))).to(device)
        out = self.activation_function(self.fcB2(out)).to(device)
        out_2 = self.activation_function(self.fcB3(out)).to(device)
        outx = torch.view_as_complex(torch.reshape(self.activation_function(self.fcB4(out_2)),(-1,self.kx,self.n_ue,2))).to(device)
        #outx = (torch.exp(1j*out)/torch.sqrt(self.kx)).to(device) # output is transformed from angle to complex number; normalize with antenna number?
        
        outy = torch.tensor([[1+0j]]).to(device)
        if self.ky>1:
            out = self.activation_function(self.fcA1(Theta)).to(device)
            out = self.activation_function(self.fcA2(out)).to(device)
            out_2 = self.activation_function(self.fcA3(out)).to(device)
            outy = torch.view_as_complex(torch.reshape(self.activation_function(self.fcA4(out_2)),(-1,self.ky,self.n_ue,2))).to(device)
            #outy = (torch.exp(1j*out)/torch.sqrt(self.ky)).to(device)
            out_allp = outx @ outy #torch.kron(outx,outy).to(device)  # Power norm so that we can compare Antenna configurations
        else:
            out_allp = outx
        #print(torch.sum(torch.abs(out_all)**2))
        #out_allp =torch.ones((len(Theta),self.kx*self.ky)).to(device) # for QAM beam comparison
        d = torch.sum(torch.sum(torch.abs(out_allp),dim=2)**2,dim=1).to(device)
        out_all = torch.reshape(out_allp,(-1,self.kx*self.ky,self.n_ue))/torch.unsqueeze(torch.unsqueeze(torch.sqrt(d),1),2)#.repeat(1,self.kx*self.ky,self.n_ue)
        #t = torch.sum(torch.abs(out_all)**2)
        
        return out_all

class Radar_receiver(nn.Module):
    """ Detects radar targets and estimates positions (angles) at which the targets are present.
    
    input paramters:
        k: number of antennas of radar receiver (linear array kx1)

    output parameters:
        detect: bool whether radar target is present
        angle: estimated angle(s)
        uncertain: uncertainty of angle estimate

     """
    def __init__(self,kx,ky,max_target=1, cpr_max=1, encoding="counting"):
        super(Radar_receiver, self).__init__()
        self.k =torch.as_tensor([kx,ky]) # Number of transmit antennas; Paper k=16
        self.detect_offset = torch.zeros((cpr_max),device=device,requires_grad=False)
        self.targetnum = max_target
        self.encoding=encoding
        self.rad_detect = Radar_detect(k=self.k,max_target=max_target, encoding=encoding).to(device)
        self.rad_angle_est = Angle_est(k=self.k,max_target=max_target).to(device)
        #self.rad_angle_est = Radar_tracking(k=self.k).to(device)
        #self.rad_angle_uncertain = Angle_est_uncertainty(k=self.k).to(device)
    
    def forward(self, c_k, targets=None, cpr=torch.tensor([1], device=device, dtype=torch.float32), noise=1):
        #cpr_max = int(torch.max(cpr))
        #cpr = torch.zeros(c_k.size()[1],1) + cpr
        detect = self.rad_detect(c_k/noise**2).to(device)
        #detect = detect/cpr #And63 -> log-likelihood scales with number of samples
        #else:
            #detect = self.rad_detect(c_k,cpr)
        Pf =0.01 # maximum error probability
        if targets!=None:
            with torch.no_grad():
                for c in range(torch.max(cpr)):
                    if (c+1) in cpr:
                        select_detect = (torch.squeeze(cpr)==c+1).to(device) # cycle for each cpr
                        #xi = torch.nonzero((1-targets))
                        xi = torch.nonzero((1-targets)*torch.unsqueeze(select_detect,1).type(torch.int16))
                        #xj = torch.nonzero(select_detect.type(torch.int16))
                        t = detect[xi[:,0],xi[:,1]]
                        sorted_nod, idx = torch.sort(torch.squeeze(t))#/noise[x[:,0]]**2)
                        off = sorted_nod[int((1-Pf)*len(sorted_nod))].to(device) # highest LLR that needs pushing to 0 (+1)
                        if idx.numel():
                            self.detect_offset[c] = off 
                
            angle_est = self.rad_angle_est(c_k,targets,cpr=cpr,noise=noise)
        else:
            if noise!=None:
                try:
                    detect = detect - self.detect_offset[cpr.type(torch.long)-1].to(device)
                except:
                    detect = detect - self.detect_offset[cpr.detach().cpu()-1].to(device)
                #ylim = torch.tensor([np.sqrt(2*np.log2(1/Pf))],device=device)*noise
                #detect = detect -ylim
            else:
                detect = detect - self.detect_offset[cpr.type(torch.long)-1]
            #detect = torch.sigmoid(detect)
            angle_est = self.rad_angle_est(c_k,cpr=cpr,noise=noise)
        
        return(detect, angle_est)#, #angle_uncertain)


class Radar_detect(nn.Module):
    def __init__(self,k,max_target,encoding):
        super(Radar_detect, self).__init__()
        self.k =torch.as_tensor(k) # Number of transmit antennas; Paper k=16
        self.d = torch.prod(k)
        self.targetnum = max_target
        #layers target_detection
        self.fcB1a = nn.Linear(self.d**2*2+1,self.d*2, device=device)
        #self.fcB1b = nn.Linear(self.d**2,self.d*2, device=device)
        #self.fcB1c = nn.Linear(1,self.d*2, device=device)
        #self.fcB1d = nn.Linear(1,self.d*2, device=device)
        self.fcB2 = nn.Linear(self.d*2,self.d*2, device=device)
        self.fcB3 = nn.Linear(self.d*2,self.d, device=device)
        if encoding=='onehot':
            self.fcB4 = nn.Linear(self.d,self.targetnum+1, device=device)
        else:
            self.fcB4 = nn.Linear(self.d,self.targetnum, device=device) # linear output layer, add one for onehot encoding
        # Non-linearity (used in transmitter and receiver) ?
        self.activation_function = nn.ELU()
    
    def forward(self, c_k,cpr=torch.tensor([[1]], device=device, dtype=torch.float32), noise=None):
        if len(c_k)!=len(cpr):
            cpr=cpr.clone().repeat(len(c_k),1)
        detect = self.target_detection(c_k, cpr, noise)
        # fix false alarm rate to 0.01 in receiver
        return detect

    def target_detection(self, c_k, cpr, noise=None):
        x_in = torch.cat((torch.real(c_k),torch.imag(c_k),cpr),1)
        out = self.activation_function(self.fcB1a(x_in.type(torch.float32))).to(device)
        #out = torch.add(out,self.activation_function(self.fcB1b(torch.imag(c_k).type(torch.float32))).to(device))
        out = self.activation_function(self.fcB2(out))
        out_2 = self.activation_function(self.fcB3(out))
        outx = (self.activation_function(self.fcB4(out_2)))
        #outx = torch.sigmoid(outx) # integrate sigmoid layer into loss function
        return(outx)

class Angle_est(nn.Module):
    def __init__(self,k,max_target=1):
        super(Angle_est, self).__init__()
        self.k =torch.as_tensor(k) # Number of transmit antennas; Paper k=16
        self.d = torch.prod(k)
        self.num_targets = max_target # prevent large network
        t=max_target
        #layers AoA est
        self.fcA1a = nn.Linear(self.d**2*2+2,self.d*8*t, device=device)
        #self.fcA1b = nn.Linear(self.d**2,self.d*4*t, device=device)
        #self.fcA1c = nn.Linear(1,self.d*4*t, device=device)
        #self.fcA1d = nn.Linear(1,self.d*4*t, device=device)
        self.fcA2x = nn.Linear(self.d*8*t,self.d*4*t, device=device)
        self.fcA2 = nn.Linear(self.d*4*t,self.d*4*t, device=device)
        #self.fcA3 = nn.Linear(self.d*4*t,self.d*t*2, device=device)
        self.fcA3 = nn.Linear(self.d*4*t,self.d*t, device=device)
        self.fcA4 = nn.Linear(self.d*t,2*max_target, device=device) # linear output layer 
        # Non-linearity (used in transmitter and receiver) ?
        self.activation_function = nn.ELU()
    
    def angle_est(self, c_k,cpr, noise=None):
        x_in = torch.cat((torch.real(c_k),torch.imag(c_k),cpr,noise),1)
        #c_k = c_k.clone()/torch.sum(torch.abs(c_k)**2,dim=0)
        out = self.activation_function(self.fcA1a(x_in.type(torch.float32))).to(device)
        out = self.activation_function(self.fcA2x(out)).to(device)
        out = self.activation_function(self.fcA2(out))
        out_2 = self.activation_function(self.fcA3(out))
        outx = (self.activation_function(self.fcA4(out_2)))
        outx = np.pi/2*torch.tanh(outx) # now two angles, elevation and azimuth
        out_all = torch.reshape(outx,(-1,2,self.num_targets))
        return(out_all)
    
    def forward(self, c_k, targets=None, cpr=1, noise=None):
        if targets==None:
            angle = self.angle_est(c_k,cpr,noise=noise)
        else:
            targ = torch.nonzero(torch.squeeze(targets))
            angle = torch.zeros((targets.size()[0],2, self.num_targets)).to(device)
            cpr = cpr[targ[:,0]]
            if noise==None:
                pass
            else:
                noise = noise[targ[:,0]]
            if targ.numel():
                angle[targ[:,0]] = self.angle_est(c_k[targ[:,0]],cpr, noise=noise)

        return angle

class Joint_radar_receiver(nn.Module):
    """ Detects radar targets and estimates positions (angles) at which the targets are present.
    
    input paramters:
        k: number of antennas of radar receiver (linear array kx1)

    output parameters:
        detect: bool whether radar target is present
        angle: estimated angle(s)
        uncertain: uncertainty of angle estimate

     """
    def __init__(self,kx,ky,max_target=1, encoding="counting"):
        super(Joint_radar_receiver, self).__init__()
        self.k =torch.as_tensor([kx,ky]) # Number of transmit antennas; Paper k=16
        self.detect_offset = 0
        self.targetnum = max_target
        self.encoding=encoding
        self.rad_detect = Radar_detect(k=self.k,max_target=max_target, encoding=encoding).to(device)
        self.rad_angle_est = Angle_est(k=self.k,max_target=max_target).to(device)
        t = max_target
        self.d = torch.prod(self.k)
        #self.rad_angle_est = Radar_tracking(k=self.k).to(device)
        #self.rad_angle_uncertain = Angle_est_uncertainty(k=self.k).to(device)
        self.fcA1a = nn.Linear(self.d**2,self.d**2, device=device)
        self.fcA1b = nn.Linear(self.d**2,self.d**2, device=device)
        self.fcA1c = nn.Linear(1,self.d**2, device=device)
        self.fcA2x = nn.Linear(self.d**2,self.d**2, device=device)
        self.fcA2 = nn.Linear(self.d**2,self.d*4*t, device=device)
        self.fcA3 = nn.Linear(self.d*4*t,self.d*t, device=device)
        self.fcA4 = nn.Linear(self.d*t,2*max_target, device=device) # linear output layer angle
        self.fcB4 = nn.Linear(self.d*t,max_target, device=device) # linear output layer detect 
        self.activation_function = nn.ELU()
    
    def forward(self, c_k, targets=None, cpr=torch.tensor([1], device=device, dtype=torch.float32)):
        cpr = torch.unsqueeze(torch.zeros(c_k.size()[0],device=device) + cpr,1)

        out = self.activation_function(self.fcA1a(torch.real(c_k).type(torch.float32))).to(device)+self.activation_function(self.fcA1b(torch.imag(c_k).type(torch.float32))).to(device)+ self.activation_function(self.fcA1c(cpr)).to(device)
        #out = self.activation_function(self.fcA1a(c_k.type(torch.float32))).to(device)
        out = self.activation_function(self.fcA2x(out))
        out = self.activation_function(self.fcA2(out))
        out_2 = self.activation_function(self.fcA3(out))
        outx = (self.activation_function(self.fcA4(out_2)))
        outx = np.pi/2*torch.tanh(outx) # now two angles, elevation and azimuth
        angle_est = torch.reshape(outx,(-1,2,self.targetnum))
        detect = (self.activation_function(self.fcB4(out_2)))
        Pf =0.01 # maximum error probability
        if targets!=None:
            x = torch.nonzero((1-targets))
            t = detect[x[:,0],x[:,1]]
            sorted_nod, idx = torch.sort(torch.squeeze(t))
            if idx.numel():
                self.detect_offset = torch.mean(sorted_nod[int((1-Pf)*len(sorted_nod))])
            detect = detect - self.detect_offset
            #angle_est = self.rad_angle_est(c_k,targets,cpr=cpr)
        else:
            detect = detect - self.detect_offset
        
        return(detect, angle_est)#, #angle_uncertain)

class Radar_tracking(nn.Module):
    """ Tracks targets and estimates positions (angles) at which the targets are present.
    Because targets move with finite speed (), a convolutional or even LSTM structure is useful.
    
    input paramters:
        k: number of antennas of radar receiver (linear array kx1)
        c_k: kxN complex values received by the radar receiver

    output parameters:
        angle: estimated angle(s)

     """
    def __init__(self,k):
        super(Radar_tracking, self).__init__()
        self.k =torch.as_tensor(k) # Number of transmit antennas; Paper k=16
        self.hidden = None
        #self.cell = None
        #layers angle tracking
        self.rnn = nn.LSTM(self.k*2,self.k*2,4, device=device)
        self.out = nn.Linear(self.k*2, 1, device=device)
    
    def angle(self, c_k):
        # calculate uncertainty
        c_k = torch.unsqueeze(c_k, dim=0)
        input1 = torch.concat((torch.real(c_k),torch.imag(c_k)),dim=2)
        #input2 = torch.concat((input1,torch.roll(input1,-1)), dim=0) # sequence is last sample
        #inputs = torch.reshape(input2,(2,c_k.size()[1],self.k*2)).type(torch.float32)
        inputs = torch.reshape(input1, (c_k.size()[1],1,self.k*2)).type(torch.float32)
        if self.hidden==None:
            out, (hn, cn) = self.rnn(inputs)
        else:
            out, (hn, cn) = self.rnn(inputs, (self.hidden, self.cell))
        self.hidden = hn
        self.cell = cn
        outx = torch.squeeze(torch.arctan(self.out(out)))*np.pi
        return(outx)
    
    def forward(self, c_k, targets=None):
        tracker = self.angle(c_k)
        return(tracker)    


def comm_loss_ra(rx,tx_bits,alpha):
    """
    weighted sum rate optimization reformulated as loss function
    max U = sum log(R_k), if apha=1,
            sum (R_k)^(1-alpha)/(1-alpha)

    input: float, shape: batch_size x M x num_ue
    R_k = (M-sum(BCE))
    """
    loss = nn.BCEWithLogitsLoss(reduction='none')
    logM = (rx[0].shape[1])

    if alpha==1:
        l=0
        for i in range(len(rx)):
            #l+= torch.mean(torch.sum(loss(rx[i],tx_bits[i]),dim=1),dim=0)
            l += -torch.log2(logM-torch.mean(torch.sum(loss(rx[i],tx_bits[i]),dim=1),dim=0))
            #l+= torch.sum(-torch.log2(1-torch.mean(loss(rx[i],tx_bits[i]),dim=0)+1e-5),dim=0)
    else:
        l=0
        for i in range(len(rx)):
            l += -torch.pow(torch.sum(logM-torch.mean(loss(rx[i],tx_bits[i]),dim=0),dim=0),1-alpha)/(1-alpha)
    return l
