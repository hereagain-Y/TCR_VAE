# calculate distance on model 1 one hot no sigmoid 

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
        self.LeakyReLU = nn.LeakyReLU(0.5)      
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
        x_hat = self.LeakyReLU(self.FC_output(h))
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

print('Read Model')
model = torch.load('/dcl02/hongkai/data/danwei/VAE/Cat_data_train_10k_VAE_64_one_AA/VAE_modified_5000_echo_cat_train.apx',map_location ='cpu')
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
#repeat 200,000 times
n = 5000
df_out = pd.DataFrame(columns=['seq1','seq2','alignment_score','latent_dist'])
for i in range(n):
    if i % 10000 == 0:
        print(i)
    seq1 = random.sample(seq, 1)[0]
    seq2 = random.sample(seq, 1)[0]
    #seq1
    mat1 = np.array(oneHotEncode(seq1))
    mat2 = AAindexEncoding(seq1)
    matrix = np.concatenate((mat1,mat2),axis=1).reshape((800,))
    matrix3 = torch.from_numpy(matrix).float()
    latent1 =  model(matrix3)[1].detach().numpy()
    #seq2
    mat3 = np.array(oneHotEncode(seq2))
    mat4 = AAindexEncoding(seq2)
    matrix = np.concatenate((mat3,mat4),axis=1).reshape((800,))
    matrix3 = torch.from_numpy(matrix).float()
    latent2 = model(matrix3)[1].detach().numpy()
    #alignments
    alignments = pairwise2.align.localds(seq1, seq2, align_matrix, open=open_penalty, extend=gap_penalty)
    score_align = alignments[0].score
    #
    latent_dist = np.sqrt( np.sum(np.square(latent1 - latent2)) )
    df_out.loc[len(df_out)] = [seq1, seq2, score_align, latent_dist]

df_out.to_csv('/dcl02/hongkai/data/danwei/VAE/Cat_data_train_10k_VAE_64_one_AA/Deepcat_data_latent_score1.csv',index=False)