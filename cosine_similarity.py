# comparing the cosine  similarity and alignment score 
# calculate distance on CNNVAE model 

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

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




cuda = False
channels =1
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 1000
hidden_dim = 256
hidden_dim2 = 128
latent_dim = 64

lr = 1e-3
epochs = 500


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


print('Read Model')
model = torch.load('/dcl02/hongkai/data/danwei/CNNVAE/Cat_data_train_500_CNN_64_AA/CNNVAE_sigmoid_modified_500_echo_cat_train.apx',map_location ='cpu')
#how to decode for one seq

#For alignment
import random
from Bio import pairwise2
from Bio.Align import substitution_matrices
#names = substitution_matrices.load()
matrix = substitution_matrices.load('BLOSUM62')
#from Bio.SubsMat import MatrixInfo as matlist
#matrix = matlist.blosum62
#in 80 max penalty for substitution is -6,
open_penalty = -4
gap_penalty = -4
#
def parseMatrix(m):
    re = {}
    alpha = m.alphabet
    for i in range(len(alpha)):
        for j in range(i,len(alpha)):
            re[(alpha[i],alpha[j])] = m[i,j]
    return re

align_matrix =  parseMatrix(substitution_matrices.load('BLOSUM62'))    
print("read data")



print("calculate distance ")
from scipy import spatial

#repeat 200,000 times
n = 5000
df_out = pd.DataFrame(columns=['seq1','seq2','alignment_score','latent_dist'])
for i in range(n):
    if i % 10000 == 0:
        print(i)
    seq1 = random.sample(seq, 1)[0]
    seq2 = random.sample(seq, 1)[0]
    #seq1
    mat2 = AAindexEncoding(seq1) # 20*20
    #matrix =mat2.reshape((800,))
    matrix3 = torch.from_numpy(mat2).float()
    # reshape the size 
    matrix3 = matrix3.view(1,1,20,20)
    latent1 =  model(matrix3)[1].detach().numpy()
    #seq2
    mat4 = AAindexEncoding(seq2)
    #matrix = np.concatenate((mat3,mat4),axis=1).reshape((800,))
    matrix4 = torch.from_numpy(mat4).float()
    matrix4 = matrix4.view(1,1,20,20)
    latent2 = model(matrix4)[1].detach().numpy()
    #alignments
    alignments = pairwise2.align.localds(seq1, seq2, align_matrix, open=open_penalty, extend=gap_penalty)
    score_align = alignments[0].score
    #cos similarity
    #latent_dist = np.sqrt( np.sum(np.square(latent1 - latent2)) )
    latent_dist = 1-spatial.distance.cosine(latent1, latent2)
    df_out.loc[len(df_out)] = [seq1, seq2, score_align, latent_dist]

df_out.to_csv('/dcl02/hongkai/data/danwei/CNNVAE/Cat_data_train_500_CNN_64_AA/Deepcat_data_latent_score1.csv',index=False)

import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

print("start plotting")
sns.scatterplot(x='alignment_score',y='latent_dist',data=df_out,s=4)
#sns.lmplot(x='alignment_score',y='latent_dist',data=distance)
r,p =stats.pearsonr(df_out['alignment_score'], df_out['latent_dist'])
print(r)
plt.text(10, 1.8, 'r={:.2f}'.format(r))
plt.savefig('/dcl02/hongkai/data/danwei/CNNVAE/Cat_data_train_500_CNN_64_AA/aligment_score.png')
