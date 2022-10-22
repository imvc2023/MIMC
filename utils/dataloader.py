import torch
import sklearn

from torch.utils.data import Dataset
import scipy
from scipy.io import loadmat

import numpy as np

def loadData(data_name):
    data = scipy.io.loadmat(data_name) 
    features = data['X']
    ## 输入维度为n*d
    if "100leaves" in data_name:
        for i in range(features.shape[1]):
            features[0][i]=np.transpose(features[0][i])
    if "ORL" in data_name:
        for i in range(features.shape[1]):
            features[0][i]=np.transpose(features[0][i])
    if "MITIndoor" in data_name:
        for i in range(features.shape[1]):
            features[0][i]=np.transpose(features[0][i])
    if "scene15" in data_name:
        for i in range(features.shape[1]):
            features[0][i]=np.transpose(features[0][i])
    if "MNIST.mat" in data_name:
        gnd = data['Y']
    else:
        gnd = data['truth']
    ## 返回一个折叠成一维的数组，只能适用于numpy对象，即array或者mat
    gnd = gnd.flatten()

    return features, gnd

class AnyDataset(Dataset):

    def __init__(self,dataname):
        self.features, self.gnd = loadData('./data/' + dataname + '.mat')
        self.v=self.features.shape[1]
        ## 数据归一化
        for i in range(0,self.v):
            minmax=sklearn.preprocessing.MinMaxScaler()
            self.features[0][i]=minmax.fit_transform(self.features[0][i])
        ## 单位矩阵
        self.iden=torch.tensor(np.identity(self.features[0][0].shape[0])).float()
        self.dataname = dataname

    ## 获得样本个数
    def __len__(self):
        return self.gnd.shape[0]

    def __getitem__(self, idx):
        '''
        return torch.from_numpy(np.array(self.features[0][:][:,idx])), torch.from_numpy(
            np.array(self.gnd[idx])), torch.from_numpy(np.array(idx))
        '''
        ## torch.from_numpy 构建张量
        if(self.v==2):
            return list([torch.from_numpy(np.array(self.features[0][0][idx],dtype=np.float32)),\
                   torch.from_numpy(np.array(self.features[0][1][idx],dtype=np.float32))]),torch.from_numpy(np.array(self.gnd[idx])), \
                   torch.from_numpy(np.array(idx)),torch.from_numpy(np.array(self.iden[idx]))
        if(self.v==3):
            return list([torch.from_numpy(np.array(self.features[0][0][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][1][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][2][idx],dtype=np.float32))]),torch.from_numpy(np.array(self.gnd[idx])), torch.from_numpy(np.array(idx)),torch.from_numpy(np.array(self.iden[idx]))
        if(self.v==4):
            return list([torch.from_numpy(np.array(self.features[0][0][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][1][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][2][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][3][idx],dtype=np.float32))]),torch.from_numpy(np.array(self.gnd[idx])), torch.from_numpy(np.array(idx)),torch.from_numpy(np.array(self.iden[idx]))
        if(self.v==5):
            return list([torch.from_numpy(np.array(self.features[0][0][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][1][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][2][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][3][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][4][idx],dtype=np.float32))]),torch.from_numpy(np.array(self.gnd[idx])),torch.from_numpy(np.array(idx)),torch.from_numpy(np.array(self.iden[idx]))
        if(self.v==6):
            return list([torch.from_numpy(np.array(self.features[0][0][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][1][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][2][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][3][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][4][idx],dtype=np.float32)),torch.from_numpy(np.array(self.features[0][5][idx],dtype=np.float32))]),torch.from_numpy(np.array(self.gnd[idx])), torch.from_numpy(np.array(idx)),torch.from_numpy(np.array(self.iden[idx]))

def dataset_with_info(dataname):
    features, gnd = loadData('./data/' + dataname + '.mat')  # The storage path of the data set
    
    views=max(features.shape[0],features.shape[1])
    input_num=features[0][0].shape[0]
    datasetforuse=AnyDataset(dataname)
    nc = len(np.unique(gnd))
    input_dims=[]
    for v in range(views):
        dim=features[0][v].shape[1]
        ## 统计每一视图的维度
        input_dims.append(dim)
    print("Data: "+dataname+", number of data: "+str(input_num)+", views: "+str(views)+", clusters: "+str(nc)+", each view: ",input_dims)

    return datasetforuse, input_num, views, nc, input_dims, gnd