# 互信息估计
import torch
import random
import sklearn
import os

import torch.nn as nn
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

from torch.autograd import Variable
from utils.dataloader import dataset_with_info, loadData
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(y_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()
    def forward(self, x, y):
        h = self.fc1(x) + self.fc2(y)
        h = self.relu(h)
        h = self.fc3(h)
        return h

def gen_x(data_size, data_dim):
    return np.sign(np.random.normal(0., 1., [data_size, data_dim]))

def gen_y(x, var = 0.2):
    return x + np.random.normal(0., np.sqrt(var), x.shape)

def do_calculate(X, Y, device="cuda:0"):
    h_dim = 50
    model = MINE(X.shape[-1], Y.shape[-1], h_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    epoch = 500
    for i in tqdm(range(epoch)):
        Y_shuffle = np.random.permutation(Y)
        
        X_train = Variable(torch.from_numpy(X).type(torch.FloatTensor), requires_grad=True)
        Y_train = Variable(torch.from_numpy(Y).type(torch.FloatTensor), requires_grad=True)
        Y_shuffle_train = Variable(torch.from_numpy(Y_shuffle).type(torch.FloatTensor), requires_grad=True)

        X_train, Y_train, Y_shuffle_train = X_train.to(device), Y_train.to(device), Y_shuffle_train.to(device)

        pred_xy = model(X_train, Y_train)
        pred_x_y = model(X_train, Y_shuffle_train)
        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = -ret
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return -1 * loss.item()
def cal_mi_orth(X, Y_list, device="cuda:0"):
    h_dim = 50
    model = MINE(X.shape[-1], Y_list[0].shape[-1], h_dim)
    model.to(device)
    ins_num = X.shape[0]
    optimizer = torch.optim.Adam(model.parameters())
    epoch = 500
    for i in tqdm(range(epoch)):
        loss = 0
        idx = random.sample(range(ins_num), ins_num)
        X_train = Variable(torch.from_numpy(X).type(torch.FloatTensor), requires_grad=True)
        for Y in Y_list:
            Y_train = Variable(torch.from_numpy(Y).type(torch.FloatTensor), requires_grad=True)
            Y_shuffle = Y[idx]
            Y_shuffle_train = Variable(torch.from_numpy(Y_shuffle).type(torch.FloatTensor), requires_grad=True)
            X_train, Y_train, Y_shuffle_train = X_train.to(device), Y_train.to(device), Y_shuffle_train.to(device)
            pred_xy = model(X_train, Y_train)
            pred_x_y = model(X_train, Y_shuffle_train)
            loss += (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))
        loss = loss / len(Y_list)
        loss = -loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return -1 * loss.item()
def cal_mi(path, device="cuda:0"):
    # 先读数据
    contents = scio.loadmat(path)
    H = contents['H']
    U = contents['U']
    S_list = []
    view_num = 0
    while True:
        if "view_" + str(view_num) not in contents.keys():
            break
        else:
            S_list.append(contents['view_' + str(view_num)])
            view_num += 1
    features, gnd = loadData('/home/zhanglei/data/multi-view/UCI.mat')
    for i in range(view_num):
        minmax=preprocessing.MinMaxScaler()
        features[0][i] = minmax.fit_transform(features[0][i])
    # 需要对数据进行归一化处理
    # 计算每个视角表征与原始数据的表征
    # mi_rec = do_calculate(features[0][2], U, device)
    mi_orth = do_calculate(H, S_list[2], device)
    # 获取k-nn的表征
    # ins_num = len(features[0][0])
    # pos_num = 21
    # neighbors_num = int(ins_num/4)
    # nbrs_idx = []
    # for v in range(view_num):
    #     X_np = np.array(features[0][v])
    #     nbrs_v = np.zeros((ins_num, pos_num - 1))
    #     nbrs = NearestNeighbors(n_neighbors=neighbors_num, algorithm='auto').fit(X_np)
    #     dis, idx = nbrs.kneighbors(X_np)
    #     for i in range(ins_num):
    #         for j in range(pos_num - 1):
    #             nbrs_v[i][j] += idx[i][j+1]
    #     nbrs_idx.append(torch.LongTensor(nbrs_v))
    # nbrs_idx = torch.cat(nbrs_idx, dim=-1)
    # nbr_embeddings_list = []
    # for i in range(pos_num - 1):
    #     nbr_idx = nbrs_idx[:, i]
    #     nbr_embeddings = U[nbr_idx]
    #     nbr_embeddings_list.append(nbr_embeddings)
    # mi_con = cal_mi_orth(U, nbr_embeddings_list)
    mi_rec = None
    mi_con = None 
    return mi_rec, mi_orth, mi_con

def plot_mi(epoches, mi, file_path, weight=1):
    plt.cla()
    plt.xlabel("Epoch", fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel("Mutual-Information", fontdict={'family': 'Times New Roman', 'size': 16})
    plt.plot(epoches, weight * np.array(mi))
    plt.savefig(file_path)
if __name__ == '__main__':
    dir_name = "./embeddings/"
    epoch_list = []
    for f in os.listdir(dir_name):
        epoch_list.append(int(f.split(".")[0]))
    epoch_list.sort()
    losses_orth = [0.000732421875, 0.00016021728515625, 0.00035858154296875, 0.00019073486328125, 6.103515625e-05, 4.1961669921875e-05, 7.62939453125e-05, 9.1552734375e-05, 3.4332275390625e-05, 3.4332275390625e-05, 1.52587890625e-05, 0.0001220703125, 0.000118255615234375, 1.9073486328125e-05, 7.62939453125e-06, 5.7220458984375e-05, 0.00019073486328125, 2.288818359375e-05, 2.288818359375e-05, 4.1961669921875e-05, 0.000186920166015625, 1.9073486328125e-05, 6.103515625e-05, 4.57763671875e-05, 3.4332275390625e-05, -2.288818359375e-05, -7.62939453125e-06, 2.6702880859375e-05, 3.4332275390625e-05, 1.71661376953125e-05, 0.0001068115234375, -1.52587890625e-05, 6.103515625e-05, 6.103515625e-05, 7.62939453125e-06, -3.814697265625e-06, 1.1444091796875e-05, -7.62939453125e-06, 0.0, 1.9073486328125e-05, 1.52587890625e-05, 7.62939453125e-06, -7.62939453125e-06, 4.9591064453125e-05, 3.0517578125e-05, 7.62939453125e-06, 7.62939453125e-06, 1.9073486328125e-05, 1.52587890625e-05, 0.0, 3.814697265625e-06, 0.0, 7.62939453125e-06, 7.62939453125e-06, 1.9073486328125e-05, 3.814697265625e-06, 9.5367431640625e-06, 1.1444091796875e-05, 0.0, -7.62939453125e-06, 7.62939453125e-06, -3.814697265625e-06, -3.814697265625e-06, 7.62939453125e-06, -7.62939453125e-06, -1.1444091796875e-05, 3.814697265625e-06, 1.1444091796875e-05, -1.52587890625e-05, 1.9073486328125e-05, -7.62939453125e-06, 0.0, -7.62939453125e-06, -3.814697265625e-06, -3.814697265625e-06, -7.62939453125e-06, 7.62939453125e-06, -3.814697265625e-06, -3.814697265625e-06, 0.0, -7.62939453125e-06, 3.814697265625e-06, -7.62939453125e-06, -7.62939453125e-06, 3.814697265625e-06, 0.0, 7.62939453125e-06, 0.0, -1.1444091796875e-05, 3.814697265625e-06, 3.814697265625e-06, 0.0, -1.52587890625e-05, 7.62939453125e-06, 3.814697265625e-06, -3.814697265625e-06, 0.0, -3.814697265625e-06, -3.814697265625e-06]
    losses_orth = np.array(losses_orth)
    epoch_list = epoch_list[2:]
    plot_mi(epoch_list, losses_orth, "./mi_imgs/mi_orth_view_2.svg", weight=10000)
