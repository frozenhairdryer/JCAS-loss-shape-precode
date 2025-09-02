#from Autoencoder.imports import *
from Autoencoder.NN_classes import * #Encoder,Decoder,Radar_receiver,Radar_detect,Angle_est
from Autoencoder.functions import *
from Autoencoder.training_routine_SNRsweep import system_runner
#import pickle
#enc =NN_classes.Encoder(M=4)
logging.info("Running autoencoder_compare.py")
#enc, dec, beam, rad_rec = pickle.load( open( "figures/trained_NNs.pkl", "rb" ) )
softmax = nn.Softmax(dim=1)

#M=torch.tensor([8])
#sigma_s=torch.tensor([0.1], dtype=float).to(device)
#sigma_c=torch.tensor([1]).to(device)
#sigma_r=torch.sqrt(1*sigma_n**2)
#K=torch.tensor([100],device=device)#/64 #*64 correction for SNR?
#sigma_n = torch.linspace(0.01*sigma_s[0],10*sigma_s[0],steps=50).to(device)

size = 30
SNR_s = 10.0**(torch.linspace(-10,10,size,device=device)/10) # dB to linear
SNR_c = 10.0**(torch.linspace(0,32,size,device=device)/10) # dB to linear
sigma_s = 1
sigma_c = 1
sigma_ns = 1/torch.sqrt(SNR_s)
sigma_nc = 1/torch.sqrt(SNR_c)


#path_list =['results/bcemod/bce.pkl','results/bcemod/bce_log2sigma.pkl']
#set_methods =["BCE","mod_BCE"]
#path_list =['set/QAM_beamcomp/001.pkl','set/QAM_beamcomp/01.pkl','set/QAM_beamcomp/09.pkl','set/QAM_beamcomp/099.pkl']
#set_methods = ['0.01','0.1','0.9','0.99']
#path_list = ['results/sweep_gmitest_09.pkl']
#set_methods = ['0.9']
#sweep =[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#path_list =['results/mimo_test.pkl']
path_list =['results/mimo_16QAM.pkl']
set_methods =["mimo"]
ws=[0.7]
num_nns = len(path_list)
m_angle_list =[]
enctype="QAM" # "NN" or "QAM"
num_ues = 2

logging.info(path_list)

M=torch.tensor([16]).to(device)
#k_a=[16,1]

N_valid = 100000
cpr=1
N_angles = N_valid
lambda_txr = 0.1
#angles = np.pi/180*(torch.linspace(-20,20,N_angles))0.1.to(device)
#angles = np.pi/180*(torch.rand(N_angles)*40-20).to(device)
mse_plot = np.zeros((num_nns+1,len(sigma_ns.detach().cpu().numpy())))
CRB_azimuth = np.zeros((num_nns,len(sigma_ns.detach().cpu().numpy())))
SNR = np.zeros((num_nns,len(sigma_ns.detach().cpu().numpy())))
prob_e_d = np.zeros((num_nns,len(sigma_ns.detach().cpu().numpy())))
prob_f_d = np.zeros((num_nns,len(sigma_ns.detach().cpu().numpy())))
prob_e_d_bm = np.zeros((num_nns,len(sigma_ns.detach().cpu().numpy())))
prob_f_d_bm = np.zeros((num_nns,len(sigma_ns.detach().cpu().numpy())))
BMI_all = np.zeros((num_nns,num_ues,len(sigma_ns.detach().cpu().numpy())))
validation_BER = np.zeros((num_nns,num_ues,len(sigma_ns.detach().cpu().numpy())))
validation_BER_bm = np.zeros((num_nns,num_ues,len(sigma_ns.detach().cpu().numpy())))
validation_SERs = np.zeros((num_nns,num_ues,len(sigma_ns.detach().cpu().numpy())))
constellation = np.zeros((2*size,num_nns, M),dtype=np.float32)
mse_n = np.zeros((N_valid))
beaml = np.zeros((num_nns+1,360))
beaml[num_nns] = np.linspace(-90,90,360)*np.pi/180 
beam_gain = np.zeros((num_nns,4))
beam_gain[:,3] = ws

biases = np.zeros((num_nns+1,size),dtype=np.float32)

for l in range(num_nns):
    if device!='cpu':
        NN_load = pickle.load( open( path_list[l], "rb" ) )
    else:
        NN_load = CPU_Unpickler( open( path_list[l], "rb")).load()
    M = NN_load[0][0].M
    system = system_runner(M,SNR_s, SNR_c, NNs= NN_load,N_valid=N_valid)
    system.N_valid = N_valid
    num_targets_trained = system.rad_rec.targetnum
    #M = system.enc[0].M
    enctype=system.enc[0].enctype
    #constellation[0::2,0] = SNR_c.detach().cpu().numpy()
    #constellation[1::2,0] = SNR_c.detach().cpu().numpy()
    symbols = torch.eye(int(M),dtype=torch.float32).to(device)
    m_angle = np.zeros(N_angles)
    m_angle_espritx = np.zeros(N_angles)

    # Initialize objects with 0
    benchmark_angle = torch.zeros(N_valid,num_targets_trained).to(device)
    benchmark_anglex =  np.zeros((N_valid,num_targets_trained))
    nn_angles = torch.zeros(N_valid,num_targets_trained).to(device)
    target_labels = torch.randint(num_targets_trained+1,(N_valid,)).to(device)
    target_ij = torch.zeros((N_valid,num_targets_trained)).to(device)
    label_tensor = torch.zeros(num_targets_trained+1,num_targets_trained).to(device)
    for x in range(num_targets_trained+1):
        label_tensor[x] = torch.concat((torch.ones(x), torch.zeros(num_targets_trained-x)))
    target_ij += label_tensor[target_labels]
    t_nums = torch.zeros(N_valid,num_targets_trained).to(device)
    t_NNx = torch.zeros(N_valid,num_targets_trained).to(device)
    angle_receiver = torch.zeros((N_valid,2), device=device)
    angle_receiver[:,0] = np.pi/180*(torch.rand(N_valid)*20+30)
    angle_receiver[:,1] = np.pi/2 # at 90 deg
    angle_receiver = angle_receiver.repeat(1,cpr).reshape(N_valid*cpr,2)

    angle_target = torch.zeros((N_valid,2,num_targets_trained), device=device)#torch.deg2rad(torch.rand(N_valid)*40-20)
    angle_target[:,0,:] += np.pi/180*(torch.rand((N_valid,num_targets_trained))*40-20).to(device)*target_ij
    angle_target[:,1,:] += np.pi/2*target_ij # at 90 deg
    angle_target_ex = angle_target.repeat(1,cpr,1).reshape(N_valid*cpr,2,num_targets_trained)
    decoded_valid=torch.zeros((N_valid*cpr,int(torch.max(M))), dtype=torch.float32, device=device)
    # Create random validation data, receiver angles and target angles
    #y_valid = torch.randint(0,int(M),(cpr*N_valid,)).to(device)
    y_valid = system.valid_labels
    for h in range(len(SNR_s)):
        with torch.no_grad(): #no gradient required on validation data
            sigma_ns_a=(torch.unsqueeze(sigma_ns,1)[h])
            sigma_nc_a=(torch.unsqueeze(sigma_nc,1)[h])

            decoded, batch_cw, t_NN, target, angle_shrunk, permuted_angle_shrunk, corr, benchmark_angle_nn, modulated  = system.run_system(train=False, cpr=[cpr,cpr], SNR_s=SNR_s[h].repeat(2), SNR_c=SNR_c[h].repeat(2))
            if enctype=="NN":
                t = torch.view_as_real(torch.squeeze(system.enc(symbols,noise=sigma_nc_a.repeat(M,1)))).detach().cpu().numpy() #
            else:
                t = torch.view_as_real(torch.squeeze(system.enc[0](torch.arange(0,int(M)).to(device)))).detach().cpu().numpy()
            constellation[2*h,l] = t[:,0]
            constellation[2*h+1,l] = t[:,1]
            
            decoded_hard=[]
            decoded_symbol=[]
            #bits=[]
            s=[]
            for i in range(num_ues):
                BMI_all[l,:,h]=torch.sum(GMI(M,decoded[i], batch_cw[i])).detach().cpu().numpy()
                decoded_hard.append(torch.round(torch.sigmoid(decoded[i])).type(torch.int16).to(device))
                decoded_symbol_h= torch.zeros((N_valid*cpr),dtype=torch.long).to(device)
                code_ints = torch.zeros(M).to(device)
                for b in range(int(torch.log2(M))):
                    decoded_symbol_h += decoded_hard[i][:,b]*2**b
                    code_ints += gray_code(M).to(device)[:,b]*2**b
                decoded_symbol.append(code_ints[decoded_symbol_h.detach().cpu()])
                #bits.append(BER(decoded_hard[i], batch_cw[i],M))
                validation_SERs[l,i,h] = SER(decoded_symbol[i], y_valid[:,i])
                validation_BER[l,i,h]= torch.mean(BER(decoded_hard[i], batch_cw[i],M))

            # detection probability
            prob_e_d[l,h] = torch.sum(torch.round(torch.squeeze(t_NN))*(torch.squeeze(target)))/torch.sum(torch.squeeze(target)).to(device)
            # false alarm rate
            prob_f_d[l,h] = torch.sum(torch.round(torch.squeeze(t_NN))*(1-torch.squeeze(target)))/torch.sum(1-torch.squeeze(target)).to(device)
         
            

            ## Angle estimation
            x_detect = torch.nonzero(t_NN*target > 0.5).to(device)[:,0] # targets that were present and were detected
            mse_plot[l,h] = torch.mean(torch.abs(torch.squeeze(angle_shrunk[x_detect,0,:])-torch.squeeze(permuted_angle_shrunk[x_detect,0,:]))**2)
            if l==0:
                mse_plot[num_nns,h] = torch.mean(torch.abs((benchmark_angle_nn[x_detect,0,:] - permuted_angle_shrunk[x_detect,0,:]))**2)

  

print('Validation BER: '  + str(validation_BER) )
logging.info('Validation BER: '  + str(validation_BER) )                

set_methods.append('ESPRIT')
#logging.debug('Validation BER after epoch %d: ' % (epoch) + str(validation_BER[epoch]) +' (loss %1.8f)' % (loss.detach().cpu().numpy()))
#logging.debug('Validation SER after epoch %d: %f (loss %1.8f)' % (epoch, validation_SERs[epoch], loss.detach().cpu().numpy()))

beam_tensor = np.pi/180*torch.tensor([[-20.0,20.0,50,70]])
direction = system.beam(beam_tensor)
phi = np.arange(-np.pi,np.pi,np.pi/180)
a_phi = radiate(system.beam.kx,torch.tensor(phi, device=device),system.beam.ky).detach().cpu().numpy().T
E_phi = np.squeeze(np.abs(direction.detach().cpu().numpy().mT @ a_phi )**2+1e-9,0).mT
E_phi_sum = np.mean((np.abs(modulated.detach().cpu().numpy() @ a_phi)**2+1e-9),0)

s = np.zeros((2+num_ues, len(phi)))
s[0] = phi*180/np.pi
s[1:num_ues+1] = E_phi.mT
s[num_ues+1] = E_phi_sum
strings = ["angles"]
for i in range(num_ues):
    strings.append("Ephi"+str(i))
strings.append("sum")
save_to_txt(s,"beampattern",strings)

# encoded = enc(y_valid_onehot)
# direction = beam(torch.deg2rad(torch.tensor([-20.0,20.0,30.0,50.0])))
# emitted = torch.kron(encoded, direction) # give the angles in which targets/receivers are to be expected)
# emitted = torch.reshape(emitted, (message_length,16))

# received, beta = rayleigh_channel(emitted, sigma_c, sigma_c, 0.1, angle_receiver)
# k = torch.arange(beam.k)+1
# channel_state = beta * torch.sum(direction * torch.exp(-1j*2*np.pi*torch.unsqueeze(k,0)*0.5*np.sin(torch.unsqueeze(angle_receiver,1))) ,axis=1)

# #atx = torch.exp(-1j*2*np.pi*torch.unsqueeze(k,0)*0.5*np.sin(torch.unsqueeze(angle_receiver,1))).T 
# #CSI = beta * torch.sum(direction * atx.T, axis=1)-18.787878 (deg) is -0.259685 (deg)

# decoded = dec(received, channel_state)
# t = GMI(M,softmax(decoded),y_valid)
# GMI_calc = torch.sum(t)
# sers = SER(softmax(decoded),y_valid)

print("GMI is: "+str(BMI_all)+" bit")
print("SER is: "+str(validation_SERs))




# sig_rad, target = radar_channel(emitted, sigma_r, sigma_n,0.1, 16, angle_target)

# detect, angle, uncertain = rad_rec(sig_rad, target)
# t_detect = target*torch.round(torch.sigmoid(torch.squeeze(detect))) # calculate rmse only if t=1 and target was detected
# x_est = torch.nonzero(t_detect)

# rmse = (torch.sqrt(torch.sum(torch.abs((angle[x_est]*np.pi - angle_target[x_est]))**2)/torch.sum(t_detect))).detach().cpu().numpy()
# sigma_theta = (np.mean((uncertain[x_est]).detach().cpu().numpy()*np.pi))
# P_detect = (torch.sum(target*torch.round(torch.sigmoid(torch.squeeze(detect))))/torch.sum(target)).detach().cpu().numpy()

print("Detection probability is: "+str(prob_e_d))
print("False alarm rate is: "+str(prob_f_d))
#print("Angle uncertainty is: "+ str(d_angle_uncertain))
#print("Angle mse is: "+ str(mse_angle_a))


""" mean_angles = np.mean(m_angle_list,axis=0)
m_angle_listx = np.transpose(np.array(m_angle_list))
plt.figure()
for l in range(num_nns):
    plt.scatter(angles.detach().cpu().numpy(),m_angle_listx[:,l], label=set_methods[l])
#plt.plot(angles.detach().cpu().numpy(),mean_angles,'--',label="Mean")
#plt.scatter(angles.detach().cpu().numpy(),m_angle_esprit,label="ESPRIT")
plt.scatter(angles.detach().cpu().numpy(),m_angle_espritx,label="ESPRIT")
#plt.fill_between(angles.detach().cpu().numpy(),m_angle-rms_bias**2, m_angle+rms_bias**2, alpha=0.2)
plt.xlabel("input angles (rad)")
plt.ylabel("Mean estimated angle (rad)")
plt.legend(loc=4)
plt.grid()
plt.savefig(figdir+"/angle_bias_compare"+namespace+".pdf") """


plt.figure()
for l in range(num_nns):
    plt.plot(10*np.log10(SNR_s.detach().cpu().numpy()),np.sqrt(mse_plot[l]), label=set_methods[l])
plt.plot(10*np.log10(SNR_s.detach().cpu().numpy()),np.sqrt(CRB_azimuth[l]), '--', label="CRB ")
plt.scatter(10*np.log10(SNR_s.detach().cpu().numpy()),np.sqrt(mse_plot[l+1]), s=2, marker='x', label=set_methods[l+1])
#plt.scatter(10*np.log10(SNR[l]),mse_plot[l+2], s=2, marker='x', label="NN2")
#plt.plot(angles.detach().cpu().numpy(),mean_angles,'--',label="Mean")
#plt.plot(angles.detach().cpu().numpy(),m_angle_esprit,label="ESPRIT")
#plt.fill_between(angles.detach().cpu().numpy(),m_angle-rms_bias**2, m_angle+rms_bias**2, alpha=0.2)
lab=[]
lab.append("SNR")
for elem in range(num_nns):
    lab.append("rmse_h"+str(elem))
for elem in range(num_nns):
    lab.append("CRB"+str(elem))
lab.append("rmse_esprit")
data = np.zeros((len(mse_plot)+num_nns+1, size))
data[0] = SNR_s.detach().cpu().numpy()
data[1:len(mse_plot)+1] = np.sqrt(mse_plot)
data[len(mse_plot)+1:] = np.sqrt(CRB_azimuth)
save_to_txt(data,"SNRsweep_rmse", label=lab)
plt.xlabel("SNR (dB)")
#plt.xlim(-10,12)
plt.ylabel("RMSE (rad)")
plt.legend(loc=3)
#plt.yscale('log')
plt.grid()
plt.savefig(figdir+"/mse_SNRsweep"+namespace+".pdf")

biases[-1] = SNR_s.detach().cpu().numpy()
save_to_txt(biases,"biases",label=set_methods[:num_nns]+["SNR"]) 

plt.figure()
for l in range(num_nns):
    plt.plot(10*np.log10(SNR_s.detach().cpu().numpy()),prob_e_d[l], label="Detect")
    plt.plot(10*np.log10(SNR_s.detach().cpu().numpy()),prob_f_d[l], label="False alarm rate")
#plt.plot(angles.detach().cpu().numpy(),mean_angles,'--',label="Mean")
#plt.plot(angles.detach().cpu().numpy(),m_angle_esprit,label="ESPRIT")
#plt.fill_between(angles.detach().cpu().numpy(),m_angle-rms_bias**2, m_angle+rms_bias**2, alpha=0.2)
plt.yscale('log')
lab=[]
lab.append("SNR")
for elem in range(num_nns):
    lab.append("detect_prob_"+str(elem))
for elem in range(num_nns):
    lab.append("detect_prob_bm"+str(elem))
for elem in range(num_nns):
    lab.append("false_alarm_"+str(elem))
for elem in range(num_nns):
    lab.append("false_alarm_bm"+str(elem))
data1 = np.zeros((4*num_nns+1, size))
data1[0] = SNR_s.detach().cpu().numpy()
data1[1:num_nns+1] = prob_e_d
data1[num_nns+1:2*num_nns+1] = prob_e_d_bm
data1[2*num_nns+1:3*num_nns+1] =prob_f_d
data1[3*num_nns+1:] =prob_f_d_bm
save_to_txt(data1,"SNRsweep_Pd", label=lab)
plt.xlabel("SNR (dB)")
#plt.xlim(-10,12)
plt.ylabel("Probability")
plt.legend(loc=3)
#plt.yscale('log')
plt.grid()
plt.savefig(figdir+"/Pd_SNRsweep"+namespace+".pdf")

plt.figure()
for l in range(num_nns):
    plt.plot(10*np.log10(1/sigma_nc.detach().cpu().numpy()**2),validation_BER[l].mT, label=set_methods[l])
plt.xlabel("SNR")
plt.ylabel("BER")
plt.savefig(figdir+"/BER_SNRsweep"+namespace+".pdf")

string_h = ["SNR"]
for i in range(num_nns):
    for j in range(num_ues):
        string_h.append("BMI"+set_methods[i]+str(j))

save1 = np.zeros((num_nns*num_ues+1,size))
save1[0] = 1/sigma_nc.detach().cpu().numpy()**2
save1[1:] = validation_BER
save_to_txt(save1,"SNRsweep_BER", label=string_h)

# save1 = np.zeros((len(set_methods),size))
# save1[0] = 1/sigma_nc.detach().cpu().numpy()**2
# save1[1:] = validation_BER_bm
# save_to_txt(save1,"SNRsweep_BER_bm", label=["SNR"]+set_methods)

plt.figure()
for l in range(num_nns):
    plt.plot(10*np.log10(1/sigma_nc.detach().cpu().numpy()**2),BMI_all[l].mT, label=set_methods[l])
plt.xlabel("SNR")
plt.ylabel("BMI")
plt.savefig(figdir+"/BMI_SNRsweep"+namespace+".pdf")

save2 = np.zeros((num_ues*num_nns+1,size))
save2[0] = 1/sigma_nc.detach().cpu().numpy()**2
save2[1:] = BMI_all

save_to_txt(save2,"SNRsweep_BMI", label=string_h)

labels = []

for s in SNR_c:
    labels.append(str(s.item())+"I")
    labels.append(str(s.item())+"Q")
for i in range(num_nns):
    g = constellation[:,i,:]
    save_to_txt(constellation[:,i,:],"SNRsweep_constellations"+set_methods[i], label=labels)
logging.info("Simulation finished!")


# BMI_vs_rmse = np.zeros((2*size,num_nns))
# for t in range(num_nns):
#     BMI_vs_rmse[0:len(SNR_c),t] = BMI_all[t]
#     BMI_vs_rmse[len(SNR_c):,t] = np.sqrt(mse_plot[t])
# labels=[]
# for i in range(size):
#     labels.append(str(SNR_s[i])+"_"+str(SNR_c[i])+"bmi")
# for i in range(size):
#     labels.append(str(SNR_s[i])+"_"+str(SNR_c[i])+"rmse")
# save_to_txt(BMI_vs_rmse,"SNRsweep_BMI_vs_rmse",label=labels)
 
save_to_txt(np.transpose(beam_gain),"ws_beamgain",label=["comm","sens","total","ws"])

l = set_methods+["angle"]
save_to_txt(beaml,"ws_beams",label=l)