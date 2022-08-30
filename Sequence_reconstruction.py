# reconstruction part 
# use decode and encode layers to reconstruct the sequence from models 
# load normalization + sigmoid VAE model 

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
        #i change it from LeakLeakyLeakyLeakyLeakyReLU 0.2 to ReLU
        self.ReLU = nn.ReLU()      
        self.training = True
        
    def forward(self, x):
        h_       = self.ReLU(self.FC_input(x))
        h_       = self.ReLU(self.FC_input2(h_))
        h_       = self.ReLU(self.FC_input3(h_))
        h_       = self.ReLU(self.FC_input4(h_))
    
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
        #i change it from LeakReLU 0.2 to ReLU
        self.ReLU = nn.ReLU(0.5)   
        
    def forward(self, x):
        h     = self.ReLU(self.FC_hidden(x))
        h     = self.ReLU(self.FC_hidden4(h))
        h     = self.ReLU(self.FC_hidden3(h))
        h     = self.ReLU(self.FC_hidden2(h))     
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
# load model 
model = torch.load('/dcl02/hongkai/data/danwei/VAE/Cat_data_train_10k_VAE_64_one_AA/VAE_Norm_sigmoid_modified_5000_echo_cat_train.apx',map_location ='cpu')
print("model loaded")

# use decode matrix 

print('Read Seq')
seq_test = pd.read_csv('/users/dyao/VAE_model/DeepcatDat/TrainingData/NormalCDR3_test.txt',delimiter='\t',header=None,names=['seq'])
seq_test['length'] = [len(seq) for seq in seq_test['seq']]
#select seqs <=20 length
seq_test = seq_test[ seq_test['length']<=20 ]
seq = list( seq_test['seq'] )


Onehot_mat = np.array( [oneHotEncode(ele) for ele in seq] )

AA_mat= GetFeatures(seq)
type(AA_mat)
# 



# combine two encoding method together 

test_data= np.concatenate((Onehot_mat,AA_mat),axis=1)
code_mat=[]
for i in range(len(AA_mat)):
    code_mat.append(np.concatenate((Onehot_mat[i],AA_mat[i]),axis=1))


seq_test_matrix = np.array(code_mat) 

# extract the decode layer without using the reparamization trick 
# GPU verison
#r1_transform=torch.from_numpy(seq_test_matrix).float() # change to tensor and float 
#m2=r1_transform.view(len(r1_transform),800)
#model(m2.cuda())[0] # original encoding layer  *orginal dim
#model(m2.cuda())[1] # the latent layer *latent_dim

# extract x-mean from encoding 
# option + shift +a 


r1_transform=torch.from_numpy(seq_test_matrix).float()
m2=r1_transform.view(len(r1_transform),800)
mat1 =model.Encoder(m2)[0] #model(m2.cuda())[1] ==latent layer 
# decoding layer
result= model.Decoder(mat1) # *800 
result2= result.view( len(result),20,40 ).detach().numpy()

mat2 = model(m2)[0]
mat2 = mat2.view( len(mat2),20,40).detach().numpy() # faltten back
print("decoded layer constructed")

# Recontruct sequence
# use result 2

# use mat2
print("start reconstruction!")
# resconstruc to sequence 
def decodeFromModel(matrix, cut=0.1):
    #previous_max = 0
    value_max = 1
    seq_decode = ''
    for i in range(len(matrix)):
        row = matrix[i]
        value_max = np.max(row)
        #if previous_max / value_max >= cut:
        #    break
        if value_max <= cut:
            continue
        seq_decode += index_code[ np.argmax(row) ]
        #previous_max = value_max   
    return seq_decode

def ReconstructSeq(matrix):
    decode_mat=[]
    for i in range(len(matrix)):
        decode_mat.append(matrix[i][:,0:20])
    seq_decode=[]
    for i in range(len(decode_mat)):
            seq_decode.append( decodeFromModel(decode_mat[i], cut=0.1) )
    return seq_decode

seq_decode1= ReconstructSeq(result2)
seq_decode2= ReconstructSeq(mat2)
df = pd.DataFrame(columns=['seq_original','seq_model1','seq_model2'])
df['seq_original'] = seq
df['seq_model1'] = seq_decode1
df['seq_model2'] = seq_decode2
print("calcutae accuarcy")
print( len(df[df['seq_original']==df['seq_model1']])/len(df) ) # 0.008
print( len(df[df['seq_original']==df['seq_model2']])/len(df) ) # 0.005 

df.to_csv('/dcl02/hongkai/data/danwei/VAE/Cat_data_train_10k_VAE_64_one_AA/test_on_deepcat_no_equal_seq.csv',index=False)

        
