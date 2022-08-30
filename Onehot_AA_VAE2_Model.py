# Onehot+AAindex VAE change Relu to LeakyReLU, 800-400-256-128-64
# modified as sigmoid + MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from six.moves import xrange
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# one hot encoded
AAs= ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
index_code = {}
code_index = {}
l_max = 20
for i in range(len(AAs)):
    index_code[i] = AAs[i]
    code_index[ AAs[i] ] = i

def oneHotEncode(seq, l_max=l_max, index_code=index_code, code_index=code_index):
    n_amino = 20
    matrix = np.zeros((l_max,n_amino)).astype(int)
    for i in range(len(seq)):
        matrix[ i , code_index[seq[i]] ] = 1
    return matrix

# pca encoded 
pca_index = pd.read_csv("/users/dyao/VAE_model/DeepcatDat/TrainingData/AA_indexPCA.csv")
d=pca_index.set_index('Unnamed: 0').T.to_dict('list')


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
seq_test = pd.read_csv('/users/dyao/VAE_model/DeepcatDat/TrainingData/NormalCDR3_test.txt',delimiter='\t',header=None,names=['seq'])
seq_test['length'] = [len(seq) for seq in seq_test['seq']]

seq_test = seq_test[ seq_test['length']<=20 ]
seq = list( seq_test['seq'] )

Onehot_mat = np.array( [oneHotEncode(ele) for ele in seq] )

AA_mat= GetFeatures(seq)
type(AA_mat)

# combine two encoding method together 

train_data= np.concatenate((Onehot_mat,AA_mat),axis=1)
len(AA_mat)
code_mat=[]
for i in range(len(AA_mat)):
    code_mat.append(np.concatenate((Onehot_mat[i],AA_mat[i]),axis=1))
code_mat[1]

seq_train_matrix = np.array(code_mat) # 20*40



r1_transform=torch.from_numpy(seq_train_matrix)
r1_transform=r1_transform.float()
train_ds, test_ds = torch.utils.data.random_split(r1_transform, (int(0.8*len(r1_transform)), len(r1_transform)-int(0.8*len(r1_transform))))
print(train_ds, test_ds)
train_loader = DataLoader(dataset=train_ds, batch_size=1000)
test_loader  = DataLoader(dataset=test_ds,  batch_size=1000)

cuda = False
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 1000
x_dim=800 # 20*20
hidden_dim = 400
hidden_dim2 = 256
hidden_dim3 = 128
latent_dim = 64

lr = 1e-3
epochs = 5000

class Encoder(nn.Module):    
    def __init__(self, input_dim, hidden_dim,hidden_dim2,hidden_dim3,latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_input3 = nn.Linear(hidden_dim, hidden_dim2)
        self.FC_input4 = nn.Linear(hidden_dim2, hidden_dim3)
        self.FC_mean  = nn.Linear(hidden_dim3, latent_dim)
        self.FC_var  = nn.Linear(hidden_dim3, latent_dim)
        #i change it from LeakLeakyLeakyLeakyLeakyLeakyReLU 0.2 to LeakyReLU
        self.ReLU = nn.ReLU(0.5)      
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        h_       = self.LeakyReLU(self.FC_input3(h_))
        h_       = self.LeakyReLU(self.FC_input4(h_))
    
        mean     = self.FC_mean(h_)  
        log_var  = self.FC_var(h_) 
        return mean, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim,hidden_dim2,hidden_dim3,output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim3)
        self.FC_hidden4 = nn.Linear(hidden_dim3, hidden_dim2)
        self.FC_hidden3 = nn.Linear(hidden_dim2, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)      
        #i change it from LeakLeakyReLU 0.2 to LeakyReLU
        self.LeakyReLU = nn.LeakyReLU(0.5)   
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden4(h))
        h     = self.LeakyReLU(self.FC_hidden3(h))
        h     = self.LeakyReLU(self.FC_hidden2(h))     
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
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

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, hidden_dim2=hidden_dim2, hidden_dim3=hidden_dim3,latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, hidden_dim2=hidden_dim2, hidden_dim3=hidden_dim3,output_dim = x_dim)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

from torch.optim import Adam
#BCE_loss = nn.BCELoss()
def loss_function(x, x_hat, mean, log_var):
    MSELoss_criterion = nn.MSELoss()
    reproduction_loss = MSELoss_criterion(x_hat, x) 
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
    plt.savefig('/dcl02/hongkai/data/danwei/VAE/train2.png')
model.train()
train_loss= []
testtoal_loss= []
for epoch in range(epochs):
    overall_loss = 0
    overall_testloss= 0
    for batch_idx, x in enumerate(train_loader):
        x = x.view(len(x), x_dim)
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
        x = x.view(len(x), x_dim)
        x = x.to(DEVICE)
        pred, mean, log_var = model(x)
        test_loss = loss_function(x, pred, mean,log_var)
        overall_testloss += test_loss.item()   
    if (epoch % 100 == 0):
      #print('====> Epoch %d done! Average Loss:  = %.2e, Average test loss = %.2e' % (epoch,overall_loss / (batch_idx*batch_size),overall_testloss/(batch_idx*batch_size)))
        
        
      print("\tEpoch", epoch , "complete!", "\tAverage Loss: ", overall_loss / len(train_ds),#overall_loss / (batch_idx*batch_size),
        "\tAverage Test Loss: " , overall_testloss/len(test_ds))
    with open('/dcl02/hongkai/data/danwei/VAE/Cat_data_train_10k_VAE_64_one_AA/record2.txt','w') as f:
      f.writelines(str(epoch))
    train_loss.append( overall_loss / len(train_ds) )
    testtoal_loss.append( overall_testloss / len(test_ds) )
    
print("Finish!!")    
print("plot curves")
plotCurve(range(1,epochs+1),train_loss,"epoch","loss",
          range(1,epochs+1),testtoal_loss,
          ['train','test'])
print('==train end===')

# save model 

torch.save(model, '/dcl02/hongkai/data/danwei/VAE/Cat_data_train_10k_VAE_64_one_AA/VAE_sigmoid_modified_5000_echo_cat_train.apx')

# save as csv 


df_loss = pd.DataFrame()
df_loss['train_loss'] = train_loss
df_loss['test_loss'] = testtoal_loss

# save model 

df_loss.to_csv('/dcl02/hongkai/data/danwei/VAE/Cat_data_train_10k_VAE_64_one_AA/sig5kmodel_error.csv',index=False)
