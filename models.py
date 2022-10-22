import torch 
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, dims, bn = False):
        super(Encoder, self).__init__()
        models = []
        for i in range(len(dims) - 1):
            models.append(nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                models.append(nn.ReLU())
            else:
                models.append(nn.Dropout(p=0.5))
                models.append(nn.Softmax())
        self.models = nn.Sequential(*models)
    def forward(self, X):
        return self.models(X)

class Project(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Project, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self,X):
        return self.fc(X)
class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        models = []
        for i in range(len(dims) - 1):
            models.append(nn.Linear(dims[i], dims[i + 1]))
            if i == len(dims) - 2:
                models.append(nn.Dropout(p=0.5))
                models.append(nn.Sigmoid())
            else:
                models.append(nn.ReLU())
        self.models = nn.Sequential(*models)
    
    def forward(self, X):
        return self.models(X)

class Discrimator(nn.Module):
    def __init__(self, in_dim):
        super(Discrimator, self).__init__()
        self.fc1 = nn.Linear(in_dim, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)
    def forward(self, X):
        out = self.relu1(self.fc1(X))
        out = self.fc2(out)
        return out

class Clustering(nn.Module):
    def __init__(self, K,d):
        super(Clustering, self).__init__()
        # input_A = 784  input_B = 784
        #self.commonz = input1
        self.weights = nn.Parameter(torch.randn(K,d).cuda(), requires_grad=True)
#        self.layer1 = nn.Linear(d, K, bias = False)

    def forward(self, comz):
        q1 = 1.0 / (1.0 + (torch.sum(torch.pow(torch.unsqueeze(comz, 1) - self.weights, 2), 2)))
        q = torch.t(torch.t(q1) / torch.sum(q1))
        loss_q = torch.log(q)
        return loss_q, q
class SSRMCVModel(nn.Module):
    def __init__(self,input_dims, ins_num, view_num, nc, h_dims=[200,100], device='cuda'):
        super().__init__()
        self.input_dims = input_dims
        self.ins_num = ins_num
        self.view_num = view_num
        self.nc = nc
        self.h_dims = h_dims
        self.device = device
        h_dims_reverse = list(reversed(h_dims))
        # 视角独有的特征编码器
        encoders_specific = []
        # 视角独有的解码器
        decoders_specific = []

        for v in range(self.view_num):
            encoder_v = Encoder([input_dims[v]] + h_dims + [nc], bn=True)
            encoder_v.to(self.device)
            encoders_specific.append(encoder_v)
            # 解码器，根据share表征以及specific表征进行解码
            decoder_v = Decoder([nc * 2] + h_dims_reverse + [input_dims[v]])
            decoder_v.to(self.device)
            decoders_specific.append(decoder_v)
        self.encoders_specific = nn.ModuleList(encoders_specific)
        self.encoders_specific.to(self.device)
        self.decoders_specific = nn.ModuleList(decoders_specific)
        self.decoders_specific.to(self.device)
        d_sum = 0
        for d in input_dims:
            d_sum += d
        # 将所有视角拼接起来进行编码
        self.encoder_share = Encoder([d_sum] + h_dims + [nc])
        self.encoder_share.to(self.device)
    
    def forward_for_hiddens(self, x_list):
        '''
        对数据进行编码 
        '''
        with torch.no_grad():
            x_total = torch.cat(x_list, dim=-1)
            hiddens_share = self.encoder_share(x_total)
            hiddens_specific = []
            hiddens = []
            for v in range(self.view_num):
                x = x_list[v]
                hiddens_specific_v = self.encoders_specific[v](x)
                hiddens_specific.append(hiddens_specific_v)
            hiddens_list = [hiddens_share] + hiddens_specific
            hiddens = torch.cat(hiddens_list, dim=-1)
            return hiddens

    def forward(self,x_list, clustering=False, target=None):
        '''
        返回
        hiddens_share: 视角共享表征
        hiddens_specific: 每个视角的specific表征，以列表形式存储
        hiddens: 每个样本的统一表征, share 拼接 各部分的specific
        recs: 每个视角的重构结果
        '''
        x_total = torch.cat(x_list, dim=-1)
        x_total = x_total.to(self.device)
        hiddens_share = self.encoder_share(x_total)
        # hiddens_share = hiddens_share - hiddens_share.mean(dim=-1).reshape((hiddens_share.shape[0], 1))
        # hiddens_share_clone = hiddens_share.clone()
        # hiddens_share = hiddens_share - hiddens_share_clone.mean(dim=0)
        recs = []
        hiddens_specific = []
        hiddens = []
        hiddens_specific_sum = 0
        for v in range(self.view_num):
            x =  x_list[v]
            hiddens_specific_v = self.encoders_specific[v](x)
            # hiddens_specific_copy = hiddens_specific_v.clone()
            # hiddens_specific_v = hiddens_specific_v - hiddens_specific_copy.mean(dim=0)
            hiddens_specific.append(hiddens_specific_v)
            hiddens_v = torch.cat((hiddens_share, hiddens_specific_v), dim=-1)
            rec = self.decoders_specific[v](hiddens_v)
            recs.append(rec)
        hiddens_list = [hiddens_share] + hiddens_specific
        hiddens = torch.cat(hiddens_list, dim=-1)
        return hiddens_share, hiddens_specific, hiddens, recs
class UncertaintyNet(nn.Module):
    """N (the number of views) FusionNet help h reconstruct x"""
    def __init__(self, h_dim):
        super(UncertaintyNet, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(h_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1))

    def forward(self, h):
        sigma = self.linears(h)
        return sigma

class MINE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))
    def forward(self,x, y):
        batch_size = x.size(0)
        tiled_x = torch.cat([x, x, ], dim=0)
        idx = torch.randperm(batch_size).long()

        shuffled_y = y[idx]
        concat_y = torch.cat([y, shuffled_y], dim=0)
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        logits = self.layers(inputs)

        pred_xy = logits[:batch_size]
        pred_x_y = logits[batch_size:]
        loss = np.log2(np.exp(1)) * (torch.mean(pred_xy) + torch.log(torch.mean(torch.exp(pred_x_y))))
        return loss