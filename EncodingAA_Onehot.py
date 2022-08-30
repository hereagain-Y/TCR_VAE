#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 17:26:07 2022

@author: hereagain
"""

import pandas as pd
import numpy as np
from numpy import argmax

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
pca_index = pd.read_csv("/Users/hereagain/Desktop/MIL/Code/DeepCAT_dat/AA_indexPCA.csv")
d=pca_index.set_index('Unnamed: 0').T.to_dict('list')
d['A']

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
seq_test = pd.read_csv('/Users/hereagain/Desktop/MIL/Code/DeepCAT_dat/TrainingData/NormalCDR3_test.txt',delimiter='\t',header=None,names=['seq'])
seq_test
seq_test['length'] = [len(seq) for seq in seq_test['seq']]

seq_test = seq_test[ seq_test['length']<=20 ]
seq = list( seq_test['seq'] )

Onehot_mat = np.array( [oneHotEncode(ele) for ele in seq] )

AA_mat= GetFeatures(seq)
type(AA_mat)

# combine two encoding method together 
#train_data= np.concatenate((Onehot_mat,AA_mat),axis=1)
len(AA_mat)
code_mat=[]
for i in range(len(AA_mat)):
    code_mat.append(np.concatenate((Onehot_mat[i],AA_mat[i]),axis=1))
code_mat[1]

seq_train_matrix = np.array(code_mat)




# method 2 : add  normalization 
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




