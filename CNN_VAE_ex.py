# the goal of this code is to add  CNN layer into the encoding layer of VAE 
# CNN -VAE using AA index only 
# 1000,1,20,20 to 64 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from six.moves import xrange
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')
# h_dim = 384 

# pca encoded 
pca_index = pd.read_csv("/users/dyao/VAE_model/DeepcatDat/TrainingData/AA_indexPCA.csv")
d=pca_index.set_index('Unnamed: 0').T.to_dict('list')

# pca normalization
data = d.items()
list_dat = list(d.values())
arr = np.array(list_dat)
ex = np.array(arr)
ex_norm = (ex-ex.min(axis=0))/(ex.max(axis=0)-ex.min(axis=0))

AAs=np.array(list(d.keys()))
new_pca = {}


for i in np.arange(20):
    new_pca[AAs[i]]=ex_norm[i]

new_pca
d= new_pca

def AAindexEncoding(Seq):
    length_seq=len(Seq)
    global l_max
    AAE=np.zeros([l_max,20])
    if length_seq<l_max:
        for amino in range(length_seq):
            AA=Seq[amino]# 
            AAE[amino,]=d[AA] # add PC value 
            
        for amino in range(length_seq,l_max):
            AAE[amino,]=np.zeros(20)
    else: 
        for amino in range(length_seq): # zero padding
            AA=Seq[amino]# 
            AAE[amino,]=d[AA]
        
    #AAE=np.transpose(AAE.astype(np.float32)) # row as PC. and column as AA sequence 
    return AAE 

  
def GetFeatures(file):
    hot_encode=[]
    for seq in file:
        hot_encode.append(AAindexEncoding(seq))
    hot_encode=np.array(hot_encode)
    result=np.array(hot_encode)
    return(result)

# read test data
seq_test = pd.read_csv('/users/dyao/VAE_model/DeepcatDat/TrainingData/NormalCDR3.txt',delimiter='\t',header=None,names=['seq'])
seq_test['length'] = [len(seq) for seq in seq_test['seq']]

seq_test = seq_test[ seq_test['length']<=20 ]
seq = list( seq_test['seq'] )

AA_mat= GetFeatures(seq)


seq_train_matrix = AA_mat # 20*40



r1_transform=torch.from_numpy(seq_train_matrix)
r1_transform=r1_transform.float()
train_ds, test_ds = torch.utils.data.random_split(r1_transform, (int(0.8*len(r1_transform)), len(r1_transform)-int(0.8*len(r1_transform))))
print(train_ds, test_ds)
train_loader = DataLoader(dataset=train_ds, batch_size=1000)
test_loader  = DataLoader(dataset=test_ds,  batch_size=1000)


cuda = False
channels =1
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 1000
hidden_dim = 256
hidden_dim2 = 128
latent_dim = 64

lr = 1e-3
epochs = 500
# input dimension ()
# duse cn to extract featurers
""" class cnn_feature(nn.Module):
    def __init__(self,channels ):
        super(cnn_feature, self).__init__()
        # cnn then flatten 
        self.conv1 = nn.Conv2d(1,32,5,1) #20-5+1=16
        self.conv2 = nn.Conv2d(32,64,3,3)# [16-3]/2+1=7 7*7
        self.conv3 = nn.Conv2d(64,128,3,1)#* 7-3+1=5 5*5 
    def forward(x) """

class Encoder(nn.Module):
    
    def __init__(self, hidden_dim, hidden_dim2,latent_dim,featureDim=64*10*10):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1,16,5,1) #20-5+1=16
        self.conv2 = nn.Conv2d(16,32,5,1)# [16-5+1=12
        self.conv3 = nn.Conv2d(32,64,3,1)#* 12-3+1 =10
        
        

        self.FC_input = nn.Linear(featureDim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_input3 = nn.Linear(hidden_dim, hidden_dim2)
        self.FC_mean  = nn.Linear(hidden_dim2, latent_dim)
        self.FC_var  = nn.Linear(hidden_dim2, latent_dim) # 64 
        
       # self.LeakyReLU = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        
        self.training = True
        
    def forward(self, x):
        h_       = self.conv1(x)
        h_       = self.conv2 (h_)
        h_       = self.conv3 (h_)
        h_       =h_.view(-1,64*10*10) # flatten 
        h_       = self.relu(self.FC_input(h_))
        h_       = self.relu(self.FC_input2(h_))
        h_       = self.relu(self.FC_input3(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                     #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim,hidden_dim2,output_dim=64*10*10):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim2) 
        self.FC_hidden3 = nn.Linear(hidden_dim2, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim) 
        self.deConv1   = nn.ConvTranspose2d(64,32,3)
        self.deConv2   = nn.ConvTranspose2d(32,16,5)
        self.deConv3   = nn.ConvTranspose2d(16,1,5)
        
             
        #i change it from LeakReLU 0.2 to ReLU
        self.ReLU = nn.ReLU()   
        
    def forward(self, x):
        h     = self.ReLU(self.FC_hidden(x))
        h     = self.ReLU(self.FC_hidden3(h))
        h     = self.ReLU(self.FC_hidden2(h))  
        h     = self.ReLU(self.FC_output(h))
        h     = h.view(-1,64,10,10)
        h     = self.ReLU(self.deConv1(h))  
        h     = self.ReLU(self.deConv2(h))  
        x_hat = torch.sigmoid(self.deConv3(h))  
        return x_hat

class Model(nn.Module):
    def __init__(self,Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean,var):  
        epsilon = torch.randn_like(var).to(DEVICE) 
        z = mean + var*epsilon
        return z
                       
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var) ]
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var

encoder = Encoder(hidden_dim=hidden_dim, hidden_dim2=hidden_dim2, latent_dim=latent_dim,featureDim=64*10*10)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, hidden_dim2=hidden_dim2, output_dim =64*10*10)
model = Model(Encoder=encoder,Decoder=decoder).to(DEVICE)


from torch.optim import Adam
BCE_loss = nn.BCELoss()
def loss_function(x, x_hat, mean, log_var):
    #MSELoss_criterion = nn.MSELoss()
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD


optimizer = Adam(model.parameters(), lr=lr)

print("start train")
def plotCurve(x_vals,y_vals,x_label, y_label,
              x2_vals=None, y2_vals=None, legend=None,figsize=(3.5,2.5)):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
    
    if legend:
        plt.legend(legend)
    plt.savefig('/dcl02/hongkai/data/danwei/CNN-VAE/train1.png')
model.train()
train_loss= []
testtoal_loss= []
for epoch in range(epochs):
    overall_loss = 0
    overall_testloss= 0
    for batch_idx, x in enumerate(train_loader):
        x = x.view(len(x),1,20,20)
        x = x.to(DEVICE)
        #zero grad
        optimizer.zero_grad()
        #----
        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean,log_var)
        overall_loss += loss.item()  
        #back prog
        loss.backward()     
        optimizer.step()
    #test loss, but not used for train
    for batch_idx, x in enumerate(test_loader):
        x = x.view(len(x),1,20,20)
        x = x.to(DEVICE)
        pred, mean, log_var = model(x)
        test_loss = loss_function(x, pred, mean,log_var)
        overall_testloss += test_loss.item()   
    if (epoch % 100 == 0):
      #print('====> Epoch %d done! Average Loss:  = %.2e, Average test loss = %.2e' % (epoch,overall_loss / (batch_idx*batch_size),overall_testloss/(batch_idx*batch_size)))
        
        
      print("\tEpoch", epoch , "complete!", "\tAverage Loss: ", overall_loss / len(train_ds),#overall_loss / (batch_idx*batch_size),
        "\tAverage Test Loss: " , overall_testloss/len(test_ds))
    with open('/dcl02/hongkai/data/danwei/VAE/Cat_data_train_10k_VAE_64_one_AA/record3.txt','w') as f:
      f.writelines(str(epoch))
    train_loss.append( overall_loss / len(train_ds) )
    testtoal_loss.append( overall_testloss / len(test_ds) )
    
print("Finish!!")    
print("plot curves")
plotCurve(range(1,epochs+1),train_loss,"epoch","loss",
          range(1,epochs+1),testtoal_loss,
          ['train','test'])
print('==train end===')