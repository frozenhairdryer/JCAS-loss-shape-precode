#from NN_classes import customloss
from imports import *
#from pykalman import KalmanFilter

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else:
            return super().find_class(module, name)


def gray_code(M):
    M = torch.tensor(int(M)).to(device)
    Mi = torch.pow(2,torch.ceil(torch.log2(M))).type(torch.int)
    code_matrix = torch.zeros((Mi,torch.ceil(torch.log2(Mi)).type(torch.int)))
    for i in range(int(torch.ceil(torch.log2(Mi)))):
        sequence = torch.concat((torch.zeros(2**i),torch.ones(2**i)))
        code_matrix[:2**(i+1),i] = sequence
        if i>0:
            v = torch.flip(code_matrix[:2**(i),:i], dims=[0])            
            code_matrix[2**(i):2**(i+1),:i] = v
    out_matrix = code_matrix[:M,:]
    return out_matrix

#contents = pickle.load(f) becomes...
#contents = CPU_Unpickler(f).load()
def PSK_encoder(idx,M):
    if max(idx)>=M or min(idx)<0:
        raiseExceptions("Modulation format not high enough")
    symbol_idx = torch.arange(M).to(device)
    symbols = torch.exp(1j*2*np.pi*symbol_idx/M)
    encoded = symbols[idx]
    return encoded

class QAM_encoder(nn.Module):
    """
    QAM Encoder of message idx (represented by symbol index) with a maximum of M different symbols
    """
    def __init__(self,M):
        super(QAM_encoder, self).__init__()
        m = int(torch.ceil(torch.sqrt(M)).to(device))
        rows = torch.arange(1,m+1,1).to(device)
        cols = torch.arange(1,m+1,1).to(device)
        mean = torch.tensor((m+1)/2).to(device)
        rows = rows-mean
        cols = cols-mean

        symbols = torch.zeros((m,m), dtype=torch.complex64).to(device)
        symbols += cols.repeat(1,m).reshape(m,m)
        symbols += 1j*rows.repeat(1,m).reshape(m,m).T
        power = torch.mean(torch.abs(symbols)**2).to(device)
        
        codex = gray_code(m).repeat(m,1).to(device)
        codey = torch.zeros_like(codex).to(device)
        for l in range(m):
            codey[l*m:(l+1)*m+1,:] = codex[l]
        #code = torch.zeros((M,int(torch.ceil(torch.log2(M))))).to(device)
        #code[:,:int(m/2)] = codex
        #code[:,:int(m/2)] = codex
        self.code = torch.concat((codex,codey),dim=1).to(device)
        self.symbols = (symbols/torch.sqrt(power)).reshape(m*m)
        self.M = M
        self.enctype="QAM"

    def forward(self,idx,noise=None):
        encoded = self.symbols[idx.detach().cpu()]
        return encoded.to(device)
    
    def coding(self):
        return self.code

class QAM_decoder(nn.Module):
    """
    QAM Encoder of message idx (represented by symbol index) with a maximum of M different symbols
    """
    def __init__(self,M):
        super(QAM_decoder, self).__init__()
        m = int(torch.ceil(torch.sqrt(M)).to(device))
        rows = torch.arange(1,m+1,1).to(device)
        cols = torch.arange(1,m+1,1).to(device)
        mean = torch.tensor((m+1)/2).to(device)
        rows = rows-mean
        cols = cols-mean

        symbols = torch.zeros((m,m), dtype=torch.complex64).to(device)
        symbols += cols.repeat(1,m).reshape(m,m)
        symbols += 1j*rows.repeat(1,m).reshape(m,m).T
        power = torch.mean(torch.abs(symbols)**2).to(device)
        
        codex = gray_code(m).repeat(m,1).to(device)
        codey = torch.zeros_like(codex).to(device)
        for l in range(m):
            codey[l*m:(l+1)*m+1,:] = codex[l]
        #code = torch.zeros((M,int(torch.ceil(torch.log2(M))))).to(device)
        #code[:,:int(m/2)] = codex
        #code[:,:int(m/2)] = codex
        self.code = torch.concat((codex,codey),dim=1).to(device)
        self.symbols = (symbols/torch.sqrt(power)).reshape(m*m)
        self.symbolsx = cols/torch.sqrt(power)
        self.symbolsy = rows/torch.sqrt(power)
        self.M = M
        self.enctype="QAM"

    def forward(self,sigIN, CSI, noise):
        mmse = torch.conj(CSI)/(torch.conj(CSI)*CSI+torch.squeeze(noise**2)).to(device)
        x_prep = sigIN*mmse # MMSE equalizer approach
        #plt.scatter(torch.real(x_prep).detach().cpu().numpy(),torch.imag(x_prep).detach().cpu().numpy())
        #plt.show()
        sigx = torch.real(x_prep)
        sigy = torch.imag(x_prep)
        x_symbol = torch.argmin(torch.abs(torch.unsqueeze(self.symbolsx,1).repeat(1,len(sigIN))-sigx)**2,axis=0)
        y_symbol = torch.argmin(torch.abs(torch.unsqueeze(self.symbolsy,1).repeat(1,len(sigIN))-sigy)**2,axis=0)
        outsymbol = x_symbol+int(torch.log2(self.M))*y_symbol
        outcode = self.code[x_symbol+int(torch.log2(self.M))*y_symbol]
        return outcode, outsymbol

class MLdecoder(nn.Module):
    def __init__(self, M, symbols, code):
        super(MLdecoder, self).__init__()
        self.M = M
        self.symbols = symbols
        self.code = code
    def forward(self, sigIN, CSI, noise):
        mmse = torch.conj(CSI)/(torch.conj(CSI)*CSI+torch.squeeze(noise**2)).to(device)
        x_prep = sigIN*mmse # MMSE equalizer approach
        symbol = torch.argmin(torch.abs(torch.unsqueeze(self.symbols,1).repeat(1,len(sigIN))-x_prep)**2,axis=0)
        outcode = self.code[symbol]
        return outcode, symbol

def SER(predictions, labels):
    """Calculates Hard decision SER
    
    Args:
    predictions (float): predicted symbol##NN autoencoder output; prediction one-hot vector for symbols
    labels (int): actually sent symbols (validation symbols)   

    Returns:
        SER (float) : Symbol error rate

    """
    #s2 = torch.argmax(predictions, 1).to(device)
    return (torch.sum( predictions!= labels))/predictions.shape[0] # Limit minimum SER to 1/N_valid with +1

def BER(predictions, valid_binary,m):
    """Calculates Hard decision bit error rate
    
    Args:
    ##predictions (float): NN autoencoder output; prediction one-hot vector for symbols
    predictions (float): NN output; predictions of bits
    m (int): number of modulation symbols per user
    labels (int): actually sent symbols (validation symbols)   

    Returns:
        ber (float) : bit error rate

    """
    # Bit representation of symbols
    #pred_binary = binaries[torch.argmax(predictions, axis=1),:]
    ber=torch.zeros(int(torch.log2(m)), device=device, requires_grad=True)
    ber = 1-torch.mean(torch.isclose(predictions.type(torch.float32), valid_binary.type(torch.float32),rtol=0.5)+0.0,axis=0, dtype=float)
    return ber

def GMI(M, my_outputs, mylabels):
    """Calculation of Generalized mutual information
    
    Args:
    M ( int): number of modulation symbols per user
    my_outputs (float): NN output logits ##Symbol probabilities
    mylabels (int): validation labels of sent symbols    

    Returns:
        r_signal: signal after channel, still upsampled

    """
    # gmi calculation
    gmi=torch.zeros(int(torch.log2(M)), device=device)
    #binaries = gray_code(M).to(device)
    #binaries = torch.tensor(cp.reshape(cp.unpackbits(cp.arange(0,M.detach().cpu().numpy(),dtype='uint8')), (-1,8)),dtype=torch.float32, device=device)
    #binaries = binaries[:,int(8-torch.log2(M).detach().cpu().numpy()):]
    #binaries[2,1] += 1
    #binaries[3,1] -= 1
    #b_labels = binaries[mylabels].int()
    # calculate bitwise estimates
    bitnum = int(torch.log2(M))
    #b_estimates = torch.zeros(len(my_outputs),bitnum, device=device)
    #P_sym = torch.bincount(mylabels)/len(mylabels)
    for bit in range(bitnum):
        #pos_0 = torch.where(binaries[:,bit]==0)[0]
        #pos_1 = torch.where(binaries[:,bit]==1)[0]
        #est_0 = torch.sum(torch.index_select(my_outputs,1,pos_0), axis=1) 
        #est_1 = torch.sum(torch.index_select(my_outputs,1,pos_1), axis=1)  # increase stability
        #print(my_outputs)
        #llr = torch.log((est_0+1e-12)/(est_1+1e-12)+1e-12) # logits are llr
        gmi[bit]=1-1/(len(mylabels))*torch.sum(torch.log2(torch.exp((2*mylabels[:,bit]-1)*(-my_outputs[:,bit]))+1+1e-12), axis=0)
        #if gmi[bit]<0:
        #    gmi[bit]=1-1/(len(mylabels))*torch.sum(torch.log2(torch.exp(-(2*b_labels[:,bit]-1)*llr)+1+1e-12), axis=0)
    return gmi.flatten()

def rayleigh_channel(sigIN, sigma_c, sigma_n, lambda_txr):
    """ Apply Rayleigh Channel Model
        parameters:
        sigIN : complex input signal into channel
        sigma_c : rayleigh fading; beta~CN(0,sigma_c^2)
        sigma_n : AWGN standard deviation
        lambda_txr : wavelength of carrier signal 
        theta_valid : Angle at which the receiver is present

        output:
        sigOUT : output signal
        beta : current fading parameter
         
    """

    beta = ((torch.randn(len(sigIN))+1j*torch.randn(len(sigIN))).to(device)*sigma_c/np.sqrt(2)).type(torch.complex64).to(device) # draw one beta value per transmission
    # sigx =  torch.sum(sigIN.T * alphTX, axis=0).type(torch.complex64)

    
    #kappa = beta #* alphTX # noch mit Energieanteil des Beamformings multiplizieren
    sigOUT = sigIN.permute(torch.arange(sigIN.ndim - 1, -1, -1)) * beta
    #sigOUT = torch.diagonal(alphTX.T @ sigIN.T)*beta
    #sigOUTa = [torch.dot(sigx[:,i], alphTX[:,i]) for i in range(len(sigIN))]
    #sigma_n = beta * sigma_n/sigma_c
    noise = torch.squeeze(sigma_n)/np.sqrt(2)*(torch.randn(sigOUT.size())+1j*torch.randn(sigOUT.size())).to(device) #add noise
    sigOUT += noise
    #SNR = 10*torch.log10(torch.mean(torch.abs(beta)**2/torch.abs(noise)**2)).to(device)
    #logging.info("Communication SNR is %f dB" % SNR)
    return torch.squeeze(sigOUT), beta 

def CFAR_detect(sigIN,N_valid, noise, Pf,cpr,beta=1):
    sig_power = torch.sum(torch.sum(torch.abs(torch.reshape(sigIN, (N_valid, cpr,-1)))**2,axis=1),axis=1)/(0.5*noise**2)
    #threshold = ncx2.ppf(1-Pf,cpr*16,beta)
    threshold =float(chi2.ppf(1-Pf,cpr*16*2))
    detect = (sig_power > threshold).type(torch.float32)
    return detect

def radar_channel_swerling1(sigIN, sigma_s, sigma_n, lambda_txr,k_antenna, phi_valid=0, target=torch.tensor([1])):
    """ Apply Rice Channel Model for the radar detection
        parameters:
        sigIN : complex input signal into channel (True antenna output without steering vectors)
        sigma_s : variance of radar cross section of target; alph~CN(0,sigma_r^2); swerling 1 model
        sigma_n : AWGN standard deviation
        lambda_txr : wavelength of carrier signal 
        theta_valid : Angle at which the radar target might be present
        k_antenna : number of antennas for radar receiver
        target : We can specify at which timesteps we want targets to appear

        output:
        sigOUT : output signal
        target (bool) : is a target present
         
    """
    d=lambda_txr/2 # distance between antennas is exaclty lambda/2
    k_dim = torch.prod(k_antenna).to(device)
    sent = (torch.zeros(sigIN.size()[0],k_dim)+0j).to(device)
    max_targ = phi_valid.size()[2]
    #alpha = (torch.zeros(sigIN.size()[0],int(max(torch.sum(target,axis=1))))+0j).to(device)
    alpha = ((torch.randn(sigIN.size()[0],max_targ).to(device)+1j*torch.randn(sigIN.size()[0],max_targ).to(device))*sigma_s/np.sqrt(2)*target).type(torch.complex64).to(device)
    #alpha = torch.zeros((sigIN.size()[0],max_targ)) + sigma_s
    for targ in range(max_targ):
        aTX_RX = torch.unsqueeze(radiate(k_antenna[0].to(device),phi_valid[:,0,targ].to(device),k_antenna[1].to(device), phi_valid[:,1,targ]),2).to(device)
        a = aTX_RX
        t = torch.matmul(a,torch.transpose(a,1,2))
        t1 = torch.matmul(t,torch.unsqueeze(sigIN,2)) 
        t2 = torch.squeeze(t1)*torch.unsqueeze(alpha[:,targ],1)
        #sent+= t2
        sent += torch.squeeze(torch.matmul(torch.matmul(aTX_RX,torch.transpose(aTX_RX,1,2)),torch.unsqueeze(sigIN,2)))*torch.unsqueeze(alpha[:,targ],1)
    sigOUT = sent+torch.unsqueeze(sigma_n,1)/np.sqrt(2)*(torch.randn((sigIN.size()[0],k_dim))+1j*torch.randn((sigIN.size()[0],k_dim))).to(device) #add noise
    #SNR = 10*torch.log10(torch.mean(torch.abs(alpha**2))/torch.mean(torch.sum(torch.abs(sigOUT)**2,axis=1))).to(device)
    #sigOUT += sent
    return torch.squeeze(sigOUT), target

def anlge_est_ML(sigIN, k_antenna=[16,1], angle_range=[-np.pi/2, np.pi/2]):
    # ML estimate based on reflected power
    a_theta = radiate(k_antenna[0],torch.linspace(angle_range[0], angle_range[1], 360),k_antenna[1]).type(torch.complex128)
    t = torch.conj(a_theta) @ sigIN.T
    _, idx = torch.max(torch.abs(t)**2, axis=0)
    estimated = (torch.linspace(angle_range[0], angle_range[1], 360)[idx]).detach().cpu().numpy()
    return estimated

def radiate(kx,phix,ky=1,phiy=torch.tensor([np.pi/2], device=device)):
    """ Creates aTX aRX (beam steering) vectors in angle direction phix for azimuth and phiy for elecation.
    Antenna array is assumed to be linear, with kx antennas horizontally and ky antennay vertically. 
    """
    ## Radiate from a kx*ky antenna array that is oriented in y-z-direction (phix is elevation, phiy is azimuth)
    kx_a=torch.arange(kx, device=device)+1
    ky_a=torch.arange(ky, device=device)+1
    radiatedx = torch.exp(1j*np.pi*torch.kron(torch.unsqueeze(kx_a,1),torch.unsqueeze(torch.sin(phix)*torch.sin(phiy),0)))#torch.unsqueeze(kx,0)*np.sin(torch.unsqueeze(phi,1))).T
    radiatedy = torch.exp(1j*np.pi*torch.kron(torch.unsqueeze(torch.cos(phiy),1),ky_a))
    radiated = (torch.unsqueeze(radiatedx.T,2) * torch.unsqueeze(radiatedy,1))
    radiated = torch.reshape(radiated, (radiated.size()[0],-1))
    return radiated

def esprit_angle_nns(x_in,n,L=torch.tensor([1],device=device),cpr=1, corr=0):
    """
    Esprit algorithm to estaimate angles from cpr samples of all N antennas
    x_in (cpr,N): input samples
    n (kx,ky): number antennas
    L : number targets, optional
    cpr : upsampling factor
    avg : averaging factor: number of calculations samples are split into for result averaging -> not used any more!
    OUT:
    angles: estimated angles in rad
    """
    N=n[0]
    if corr!=1:
        i_ss = int(cpr*n[1])
        #x_in = torch.squeeze(x_in)
        if i_ss>1:
            x_i = torch.reshape(x_in, (cpr, n[0], n[1]))
            x_i = torch.transpose(x_i, 1,2)
            x_i = torch.transpose(x_i, 1,0)
            x_i = torch.transpose(torch.reshape(x_i,(cpr*n[1],n[0])),0,1)
        N = n[0]
        angles = torch.zeros(torch.max(L))
        if i_ss == 1:
            x_i = torch.reshape(x_in,(n[0],i_ss))
            #x_ij = x_i[:,h*i_ss:(h+1)*i_ss]
            ##l = int(torch.round(n[0]/4+n[0]/6))
            N=n[0]-L
            f = [x_i[idx:n[0]+idx-L] for idx in range(L+1)]
            x_ij = torch.squeeze(torch.stack(f,axis=1)) # single snapshot ESPRIT: create Hankel Matrix to prevent ambiguity
            ##x_ij = torch.stack(f,axis=1)
        else:
            x_ij = x_i
        R = x_ij @ torch.transpose(torch.conj(x_ij),0,1)
    else:
        R = x_in
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    _,U = torch.linalg.eig(R)
    S = U[:,0:L]
    Phi = torch.linalg.pinv(S[0:N-1]) @ S[1:N] # the original array is divided into two subarrays [0,1,...,N-2] and [1,2,...,N-1]
    eigs, _ = torch.linalg.eig(Phi)
    DoAsESPRIT = torch.arcsin(torch.angle(eigs)/np.pi)
    angles = DoAsESPRIT
    return angles


# Saving to output file
def save_to_txt(x_in,s="name", label='signal'):
    result = [
                l+" " for l in label
    ]
    result.append("\n")
    x_in = np.transpose(x_in)
    for l in x_in:
        for t in l:
            result.append(str(t)+" "
                        )
        result.append("\n")
    with open(figdir+'/'+str(s)+'.txt'.format(), 'w') as f:
        f.writelines(result)


def plot_training(SERs,BERs,M, const,  gmi_exact, direction, detect_error, d_angle,benchmark = None, stage=None, namespace="", CRB=0, antennas=torch.tensor([16,1], device=device), enctype="NN"):
    """Creates mutliple plots in /figures for the best implementation
    
    Args:
    SERs (float): Hard-decision Symbol error rates
    valid_r (float, float) : Received signal (decoder input) 
    cvalid (int) : integer specifying symbol number (range is 0...M_all) -> colorcoded symbols 
    M (int): number of modulation symbols per user
    const (complex), len=M_all : resulting constellation (channel input)
    GMIs_appr (float) : GMI estimate from SERs
    decision_region_evolution (int) : grid containing ints denoting the corresponding symbol
    meshgrid (float): grid on which decision_region_evolution is based
    constellation_base (complex) [len(M)]: contains all possible outputs for all encoders
    gmi_exact (float): GMI calculated from LLRs -> exact value 
       
    Plots:
     * SERs vs. epoch
     * GMIs (gmi_exact) vs. epoch
     * scatterplot const
     * scatterplot valid_r as complex number
     * scatterplot complex decision regions together with received signal
     * scatterplots for base constellations

    Returns:
        none

    """
    # matplotlib.rcParams.update({
    # "pgf.texsystem": "pdflatex",
    # 'font.family': 'serif',
    # 'text.usetex': True,
    # 'pgf.rcfonts': False,
    # 'font.size' : 10,
    # 'figure.max_open_warning' : 100
    # })
    #GMIs_appr, decision_region_evolution, meshgrid,
    #plt.set_visible(False)
    cmap = matplotlib.cm.tab20
    base = plt.get_cmap(cmap)
    color_list = base.colors
    new_color_list = np.array([[t/2 + 0.49 for t in color_list[k]] for k in range(len(color_list))])

    min_SER_iter = np.argmin(SERs,axis=0)
    max_GMI = len(SERs)-1 #last iter   #np.argmax(np.sum(gmi_exact, axis=1))
    ext_max_plot = 1.2#*np.max(np.abs(valid_r[int(min_SER_iter)]))

    print('Minimum mean SER obtained:' + str(np.min(SERs,axis=0)) + '(epochs ' + str(min_SER_iter) + ' out of %d)' % len(SERs))
    print('Maximum obtained GMI: %1.5f (epoch %d out of %d)' % (np.sum(gmi_exact[max_GMI]),max_GMI,len(SERs)))
    print('The corresponding constellation symbols are:\n', const)

    plt.figure("SERs "+str(stage),figsize=(3.5,3.5))
    #plt.figure("SERs",figsize=(3.5,3.5))
    plt.plot(SERs,marker='.',linestyle='--',markersize=2)
    plt.plot(min_SER_iter,np.min(SERs,axis=0),marker='o',markersize=3,c='red')
    #plt.annotate('Min', (0.95*min_SER_iter,1.4*SERs[min_SER_iter]),c='red')
    plt.xlabel('epoch no.')
    plt.ylabel('SER')
    plt.grid(visible=True,which='both')
    #plt.legend(loc=1)
    plt.title('SER on Validation Dataset')
    #plt.tight_layout()
    #tikzplotlib.clean_figure()
    plt.savefig(figdir+"/Sers"+str(stage)+namespace+".pdf")
    #tikzplotlib.save("figures/SERs.tex", strict=True, externalize_tables=True, override_externals=True)

    plt.figure("BERs "+str(stage),figsize=(3.5,3.5))
    #plt.figure("SERs",figsize=(3.5,3.5))
    plt.plot(torch.mean(BERs,axis=1).detach().cpu().numpy(),marker='.',linestyle='--',markersize=2)
    #plt.annotate('Min', (0.95*min_SER_iter,1.4*SERs[min_SER_iter]),c='red')
    plt.xlabel('epoch no.')
    plt.ylabel('BER')
    plt.grid(visible=True,which='both')
    #plt.legend(loc=1)
    plt.title('SER on Validation Dataset')
    #plt.tight_layout()
    #tikzplotlib.clean_figure()
    plt.savefig(figdir+"/Sers"+str(stage)+namespace+".pdf")
    #tikzplotlib.save("figures/SERs.tex", strict=True, externalize_tables=True, override_externals=True)

    for i in range(gmi_exact.shape[2]):
        plt.figure("GMIs "+str(stage)+str(i),figsize=(3,2.5))
        #plt.plot(GMIs_appr.cpu().detach().numpy(),linestyle='--',label='Appr.')
        #plt.plot(gmi_hd,linestyle='--',label='GMI Hard decision')
        #plt.plot(max_GMI,GMIs_appr[max_GMI],c='red')
        for num in range(len(gmi_exact[0,:])):
            if num==0:
                t=gmi_exact[:,num,i]
                plt.fill_between(np.arange(len(t)),t, alpha=0.4)
            else:
                plt.fill_between(np.arange(len(t)),t,(t+gmi_exact[:,num,i]),alpha=0.4)
                t+=gmi_exact[:,num,i]
        plt.plot(t, label='GMI')
        plt.plot(np.argmax(t),max(t),marker='o',c='red')
        plt.annotate('Max', (0.95*np.argmax(t),0.9*max(t)),c='red')
        plt.xlabel('epoch no.')
        plt.ylabel('GMI')
        #plt.ylim(0,np.round(np.max(t))+1)
        #plt.xlim(0,len(t)-1)
        #plt.legend(loc=3)
        plt.grid(visible=True,which='both')
        plt.title('GMI on Validation Dataset')
        #plt.tight_layout()
        plt.savefig(figdir+"/gmis"+str(stage)+namespace+str(i)+".pdf")
        #tikzplotlib.save("figures/gmis.tex", strict=True, externalize_tables=True, override_externals=True)

    try:
        constellations = np.asarray(const)
    except:
        constellations = np.asarray(const).get()
    bitmapping=[]
    helper= np.arange((int(torch.prod(M))))
    #enctype="NN"
    if enctype=="NN":
        #for h in helper:
        #    if M==16:
        #        bitmapping.append(format(h, ('04b')))
        #    elif M==8:
        #        bitmapping.append(format(h, ('03b')))
        #    else:
        #        
        #        str_b = '0' + str(t) + 'b'
        #        bitmapping.append(format(h, (str_b)))
        bitmapping=[]
        t = int(torch.log2(M).detach().cpu().numpy())
        #code_ints = torch.zeros(M,dtype=torch.long).to(device)
        #for b in range(int(torch.log2(M))):
        #    code_ints += (gray_code(M).to(device)[:,b]*2**b).type(torch.long)
        bits = gray_code(M).to(device)
        for h in helper:
            bitmapping.append("".join(str(int(bits[h,i].detach().cpu().numpy())) for i in range(t)))

    else:
        enc = QAM_encoder(M).to(device)
        mapping = enc.coding()
        #_ , mapping =QAM_encoder(torch.zeros(3,dtype=torch.long).to(device),M,encoding=True)
        bitmapping=[]
        for l in range(mapping.size()[0]):
            bitmapping.append("".join(str(int(mapping[l,i].detach().cpu().numpy())) for i in range(mapping.size()[1])))

    for s in range(constellations.shape[1]):
        plt.figure("constellation "+str(s)+str(stage), figsize=(5,5))
        #plt.subplot(121)
        plt.scatter(np.real(constellations[:,s]),np.imag(constellations[:,s]),c=range(int(M)), cmap='tab20',s=50)
        for i in range(len(constellations)):
            plt.annotate(bitmapping[i], (np.real(constellations)[i,s], np.imag(constellations)[i,s]))
        
        plt.axis('scaled')
        plt.xlabel(r'$\Re\{r\}$')
        plt.ylabel(r'$\Im\{r\}$')
        plt.xlim((-1.5,1.5))
        plt.ylim((-1.5,1.5))
        plt.grid(visible=True,which='both')
        #plt.title('Constellation')
        #plt.tight_layout()
        plt.savefig(figdir+"/constellation"+str(stage)+namespace+".pdf")
        t = np.zeros((3,len(constellations)))
        t[0] = np.real(constellations[:,s])
        t[1] = np.imag(constellations[:,s])
        t[2] = bitmapping
        save_to_txt(t,s="constellation"+str(s),label=["real","imag","bitmapping"])
    #tikzplotlib.save("figures/constellation.tex", strict=True, externalize_tables=True, override_externals=True)

    

    # plt.figure("Received signal"+str(stage),figsize=(2.7,2.7))
    # #plt.subplot(122)
    # plt.scatter(np.real(val_cmplx[0:1000]), np.imag(val_cmplx[0:1000]), c=cvalid[0:1000,0].cpu().detach().numpy(), cmap='tab20',s=2)
    # plt.axis('scaled')
    # plt.xlabel(r'$\Re\{r\}$')
    # plt.ylabel(r'$\Im\{r\}$')
    # plt.xlim((-2,2))
    # plt.ylim((-2,2))
    # plt.grid(visible=True)
    # plt.title('Received')
    # #plt.tight_layout()
    # plt.savefig(figdir+"/received"+str(stage)+namespace+".pdf")
    #tikzplotlib.save("figures/received.tex", strict=True, externalize_tables=True, override_externals=True)

    
    
    
    # plt.figure("Decision regions"+str(stage), figsize=(5,3))
    # for num in range(constellations.shape[1]):
    #     plt.subplot(1,constellations.shape[1],num+1)
    #     decision_scatter = decision_region_evolution
    #     grid=np.asarray(meshgrid)
    #     if num==0:
    #         plt.scatter(grid[:,0], grid[:,1], c=decision_scatter[num],s=2,cmap=matplotlib.colors.ListedColormap(colors=new_color_list[0:int(M)]))
    #     else:
    #         plt.scatter(grid[:,0], grid[:,1], c=decision_scatter[num],s=2,cmap=matplotlib.colors.ListedColormap(colors=new_color_list[0:int(M)]))
    #     #plt.scatter(validation_received[min_SER_iter][0:4000,0], validation_received[min_SER_iter][0:4000,1], c=y_valid[0:4000], cmap='tab20',s=4)
    #     plt.scatter(np.real(val_cmplx[0:1000]), np.imag(val_cmplx[0:1000]), c=cvalid[0:1000,num].cpu().detach().numpy(), cmap='tab20',s=2)
    #     plt.axis('scaled')
    #     #plt.xlim((-ext_max_plot,ext_max_plot))
    #     #plt.ylim((-ext_max_plot,ext_max_plot))
    #     plt.xlim((-2,2))
    #     plt.ylim((-2,2))
    #     plt.xlabel(r'$\Re\{r\}$')
    #     plt.ylabel(r'$\Im\{r\}$')
    #     plt.title('Decoder %d' % (num+1))
    # #plt.tight_layout()
    # #tikzplotlib.clean_figure()
    # plt.savefig(figdir+"/decision_regions"+str(stage)+namespace+".pdf")
    # #tikzplotlib.save("figures/decision_regions.tex", strict=True, externalize_tables=True, override_externals=True)

    #a_phi = np.exp(-1j*np.pi*np.sin(np.arange(30)/30*2*np.pi))
    #al_phi = a_phi[:]**(np.arange(np.shape(encoded)[1]))
    #enc =np.array(encoded)
    phi = np.arange(-np.pi,np.pi,np.pi/180)
    theta = np.arange(0,np.pi,np.pi/90)
    #a_phi = np.zeros((30,np.shape(encoded)[1]))

    #a_phi = np.tile(np.exp( -1j*np.pi*np.sin(phi)),(np.shape(enc)[2],1))
    #a_phix = (a_phi.T**(np.arange(np.shape(enc)[2])+1)).T

    
    a_phi = radiate(antennas[0],torch.tensor(phi, device=device),antennas[1]).detach().cpu().numpy().T
    a_theta = radiate(antennas[0],torch.tensor([0], device=device),antennas[1],torch.tensor(theta, device=device)).detach().cpu().numpy().T

    plt.figure("Beampattern Horizontal", figsize=(5,3))
    #for d in range(len(encoded)):
    #enc1 = np.array(encoded)[0]
    #E_phi = 10*np.log10(np.mean(np.abs(enc1 @ a_phi )**2, axis=0))
    E_phi = np.squeeze(10*np.log10(np.abs(direction.detach().cpu().numpy().mT @ a_phi )**2+1e-9),0).mT
    E_phi2 = np.squeeze(10*np.log10(np.abs(direction.detach().cpu().numpy().mT @ a_theta )**2+1e-9),0).mT
    
    plt.plot(phi*180/np.pi, E_phi,label='epoch '+str(max_GMI)+" stage "+str(stage))
    plt.xlabel("Angle (deg) horizontal")
    plt.ylabel("radiated Power (dB)")
    plt.xlim(-90,90)
    #plt.ylim(-40,20)
    plt.legend()
    plt.grid(visible=True)
    plt.savefig(figdir+"/E_phi_h"+namespace+".pdf")

    s = np.zeros((1+SERs.shape[1], len(phi)))
    s[0] = phi*180/np.pi
    s[1:] = E_phi.mT
    strings = ["angles"]
    for i in range(SERs.shape[1]):
        strings.append("Ephi"+str(i))
    save_to_txt(s,"beampattern",strings)

    plt.figure("Beampattern Azimuth", figsize=(5,3))
    plt.plot(theta*180/np.pi, E_phi2,label='epoch '+str(max_GMI)+" stage "+str(stage))
    plt.xlabel("Angle (deg) azimuth")
    plt.ylabel("radiated Power (dB)")
    plt.xlim(0,180)
    #plt.ylim(-40,20)
    plt.legend()
    plt.grid(visible=True)
    plt.savefig(figdir+"/E_phi_a"+namespace+".pdf")

    ### Animation Beamforming
    #from matplotlib.animation import FuncAnimation
    #plt.style.use('seaborn-pastel')
    

    """ filenames = []
    for d in range(len(encoded)):
        enc1 = np.array(encoded)[d]
        E_phi = 10*np.log10(np.mean(np.abs(enc1 @ a_phi )**2, axis=0))
        # plot the line chart
        plt.figure(figsize=(6,4))
        plt.plot(phi*180/np.pi, E_phi,label='epoch no. '+str(d))
        plt.fill_between(phi[160:201]*180/np.pi,E_phi[160:201],-50,alpha=0.2,label='radar target')
        plt.fill_between(phi[210:231]*180/np.pi,E_phi[210:231],-50,alpha=0.2,label='receiver')
        plt.xlabel("Angle (deg)")
        plt.ylabel("radiated Power (dB)")
        plt.xlim(-90,90)
        plt.ylim(-40,5)
        plt.legend(loc=3)
        plt.grid(visible=True)
        plt.tight_layout()
        
        # create file name and append it to a list
        filename = figdir+f'/{d}.png'
        filenames.append(filename)
        
        # save frame
        plt.savefig(filename)
        plt.close()# build gif

    with imageio.get_writer(figdir+"/beamform"+str(stage)+namespace+".gif", mode='I', fps=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        
    # Remove files
    for filename in set(filenames):
        os.remove(filename) """

    ###
    
    plt.figure("detection error")
    #plt.figure("SERs",figsize=(3.5,3.5))
    plt.plot(detect_error[0],marker='.',linestyle='--',markersize=2, label="detection prob "+str(stage))
    plt.plot(np.argmax(detect_error[0]),np.max(detect_error[0]),marker='o',markersize=3,c='red')
    plt.annotate('Max', (np.argmax(detect_error[0]),np.max(detect_error[0])),c='red')

    plt.plot(detect_error[1],marker='.',linestyle='--',markersize=2, label="False alarm rate "+str(stage))
    plt.plot(np.argmin(detect_error[1]),np.min(detect_error[1]),marker='o',markersize=3,c='red')
    #plt.annotate('Min', (np.argmin(detect_error[0]),np.min(detect_error[0])),c='red')
    plt.xlabel('epoch no.')
    plt.ylabel(r'$P$')
    plt.grid(visible=True,which='both')
    plt.legend(loc=2)
    #plt.legend(loc=1)
    plt.title('Detection probability on Validation Dataset')
    #plt.tight_layout()
    #tikzplotlib.clean_figure()
    plt.savefig(figdir+"/PeDetect"+namespace+".pdf")
    
    
    try:
        plt.figure("Angle estimate",figsize=(6,3.5))
        x=np.arange(len(d_angle))
        plt.plot(x,np.abs(d_angle), label="cycle"+str(stage))
        try:
            plt.plot(x,np.abs(benchmark), label="ESPRIT")
        except:
            pass
        plt.plot(x,np.repeat(np.sqrt(CRB[0]),len(d_angle)),'--',label="CRB")
        plt.xlabel('epoch no.')
        plt.ylabel(r"RMSE (rad)")
        plt.grid(visible=True)
        #plt.ylim(0,0.13)
        plt.xlim(0,len(d_angle)-1)
        plt.yscale('log')
        plt.legend(loc="upper left")
        #plt.tight_layout()
        plt.savefig(figdir+"/Angle_est"+namespace+".pdf")
        saved = np.zeros((3,len(d_angle)))
        saved[0] = x
        saved[1] = np.abs(d_angle)
        saved[2] = np.repeat(np.sqrt(CRB[0]),len(d_angle))
        save_to_txt(saved,s="RMSEangle",label=["epoch","RMSE","CRB"])
    except:
        pass

    """ try:
        plt.figure("detection vs SER")
        #plt.figure("SERs",figsize=(3.5,3.5))
        plt.scatter(detect_error[0],SERs)
        
        plt.xlabel(r'$P$ '+str(stage))
        plt.ylabel("SER")
        plt.yscale('log')
        plt.grid(which='both')
        plt.legend(loc=1)
        plt.ylim(1e-3,1)
        #plt.legend(loc=1)
        plt.title('Detection probability vs SER')
        plt.tight_layout()
        #tikzplotlib.clean_figure()
        plt.savefig("figures/Detect_vs_SER.pdf")
    except:
        pass """

    

