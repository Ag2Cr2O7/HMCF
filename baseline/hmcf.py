import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from reckit import randint_choice
from manifold.hyperboloid import Hyperboloid
import math


class hmcf(nn.Module):
    def __init__(self, data_config, args):
        super(hmcf, self).__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.plain_adj = data_config['plain_adj']
        self.all_h_list = data_config['all_h_list']
        self.all_t_list = data_config['all_t_list']
        self.A_in_shape = self.plain_adj.tocoo().shape # (u+i,u+i)
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).cuda()
        self.D_indices = torch.tensor([list(range(self.n_users + self.n_items)), list(range(self.n_users + self.n_items))], dtype=torch.long).cuda()
        self.all_h_list = torch.LongTensor(self.all_h_list).cuda()
        self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
        self.G_indices, self.G_values = self._cal_sparse_adj()
        self.emb_dim = args.embed_size
        self.n_layers = args.n_layers
        self.temp = args.temp
        self.manifold = Hyperboloid()
        self.batch_size = args.batch_size
        self.emb_reg = args.emb_reg
        self.ssl_reg = args.ssl_reg
        self.dropout=args.dropout
        self.zu,self.zuu,self.zi,self.zii,self.hu,self.hi=[None] * self.n_layers,[None] * self.n_layers,[None] * self.n_layers,[None] * self.n_layers,[None] * self.n_layers,[None] * self.n_layers
        self.lsvd=args.lhyper
        self.hweight=100
        self.htemp=args.temp
        self.c = 1
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        self.suser_embedding=nn.Embedding(self.n_users, self.emb_dim)
        self.sitem_embedding=nn.Embedding(self.n_items, self.emb_dim)
        self.zerou = torch.zeros(self.n_users, 1)
        self.zeroi = torch.zeros(self.n_items, 1)
