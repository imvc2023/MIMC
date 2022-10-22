import random
import math
import torch
import scipy
import os
import json

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from utils.dataloader import dataset_with_info
from models import SSRMCVModel, Discrimator
from loss import MI
from utils import Logger
from sklearn.cluster import KMeans 
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.clusteringPerformance2 import clusteringMetrics
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE

seed = 0
scipy.random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(seed) 
os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
def plot_loss(losses, losses_dir, datasetname):
    
    plt.xlabel("Epoch", fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel("Total Loss", fontdict={'family': 'Times New Roman', 'size': 18})
    plt.plot(losses)
    figpath = losses_dir + datasetname + ".svg"
    plt.savefig(figpath)
    # 保存json文件
    with open(losses_dir + datasetname + ".json", 'w+') as fp:
        json.dump({"data":losses}, fp)
    return figpath
def plot_embeddings(datas, labels, path, epoch):
    datas_min, datas_max = datas.min(0), datas.max(0)
    datas_norm = (datas - datas_min) / (datas_max - datas_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.scatter(datas_norm[:, 0], datas_norm[:, 1], c=labels, cmap='tab10', marker='.')
    plt.xticks([])
    plt.yticks([])
    if not os.path.exists(path):
        os.makedirs(path)
    svg_path = path + "/" + str(epoch) + ".svg"
    plt.savefig(svg_path)
    return svg_path
def orthogonal_loss(shared, specific):
    _shared = shared.detach()
    _shared = _shared - _shared.mean(dim=0)
    correlation_matrix = _shared.t().matmul(specific)
    norm = torch.norm(correlation_matrix, p=1)
    return norm
def neg_sample(idx, pos_idx):
    ret = []
    for i in idx:
        if i not in pos_idx:
            ret.append(i)
    return ret
if __name__ == '__main__':
    datasetname = "UCI"
    logger = Logger.get_logger(__file__, datasetname)
    datasetforuse, ins_num, view_num, nc, input_dims, gnd = dataset_with_info(datasetname)
    train_loader = DataLoader(datasetforuse, batch_size=10000, shuffle=False)
    test_loader = DataLoader(datasetforuse, batch_size=10000, shuffle=False)
    train_epoch = 500
    ACC_array = np.zeros((6, 6))
    NMI_array = np.zeros((6, 6))
    Purity_array = np.zeros((6, 6))
    ARI_array = np.zeros((6, 6))
    params = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    # alpha
    index_mi = 3
    # beta
    index_con = 2
    lambda_rec = 1 
    lambda_mi = params[index_mi]
    lambda_con = params[index_con]
    print("===============start training===============")
    print("param ", "alpha:", lambda_mi, "beta:",lambda_con)
    do_contrast = True
    feature_dim = 64
    device="cuda:0"
    model = SSRMCVModel(input_dims, ins_num, view_num, nc=feature_dim, device=device, h_dims=[500,200])
    lr=0.001
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    clustering_epoch = 20
    # 聚类中心
    centers = None
    target = None
    q = None
    neighbors_num = int(ins_num/4)

    pos_num = 21
    neg_num = int((neighbors_num - pos_num - 1) / 2)
    nbrs_idx = []
    neg_idx = []
    losses = []
    for v in range(view_num):
        X_np = np.array(datasetforuse.features[0][v])
        nbrs_v = np.zeros((ins_num, pos_num - 1))
        neg_v = np.zeros((ins_num, neg_num))
        nbrs = NearestNeighbors(n_neighbors=neighbors_num, algorithm='auto').fit(X_np)
        dis, idx = nbrs.kneighbors(X_np)
        for i in range(ins_num):
            for j in range(pos_num - 1):
                nbrs_v[i][j] += idx[i][j+1]
            for j in range(neg_num):
                neg_v[i][j] += idx[i][neighbors_num - j - 1]
        nbrs_idx.append(torch.LongTensor(nbrs_v))
        neg_idx.append(torch.LongTensor(neg_v))
    nbrs_idx = torch.cat(nbrs_idx, dim=-1)
    neg_idx = torch.cat(neg_idx, dim=-1)
    losses = []
    for epoch in range(train_epoch):
        save_loss = True
        for x, _, idx, inpu in train_loader:
            optimizer.zero_grad()
            model.train()
            for v in range(view_num):
                x[v] = x[v].to(device)
            clustering = epoch > clustering_epoch
            hiddens_share, hiddens_specific, hiddens, recs = model(x, clustering=False)

            loss_rec = 0
            loss_mi = 0
            loss_d = 0
            loss_g = 0 
            labels_true = torch.ones(x[0].shape[0]).long().to(device)
            labels_false = torch.zeros(x[0].shape[0]).long().to(device)
            for v in range(view_num):
                # 重构损失
                loss_rec += lambda_rec * mse_loss_fn(recs[v], x[v])
                # 正交损失
                loss_mi += lambda_mi * orthogonal_loss(hiddens_share, hiddens_specific[v])
            
            loss_con = 0
            if do_contrast:
                for i in range(len(idx)):
                    # 正例相似度
                    index = idx[i]
                    
                    hiddens_positive = hiddens[nbrs_idx[index]]
                    positive = torch.exp(torch.cosine_similarity(hiddens[i].unsqueeze(0), hiddens_positive.detach()))
                    negative_idx = neg_idx[index]
                    hiddens_negative = hiddens[negative_idx]
                    negative = torch.exp(torch.cosine_similarity(hiddens[i].unsqueeze(0), hiddens_negative.detach())).sum()
                    loss_con -= lambda_con * torch.log((positive / negative)).sum()
                loss_con /= len(idx)
            loss = loss_rec + loss_mi + loss_con
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        # 进行测试
        with torch.no_grad():
            for x, _, idx, inpu in test_loader:
                for v in range(view_num):
                    x[v] = x[v].to(device)
                model.eval()
                hiddens_share, hiddens_specific, hiddens, recs = model(x)
                kmeans = KMeans(n_clusters=nc,n_init=50)
                datas = hiddens.clone().cpu().numpy()
                y_pred = kmeans.fit_predict(datas)
                ACCo, NMIo, Purityo, ARIo, Fscoreo, Precisiono, Recallo=clusteringMetrics(gnd,y_pred)  
                info = {"epoch": epoch, "acc": ACCo, "nmi": NMIo, "ari": ARIo, "Purity": Purityo, "fscore": Fscoreo, "percision": Precisiono, "recall": Recallo}
                logger.info(str(info))