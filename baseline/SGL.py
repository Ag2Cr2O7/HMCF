import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from reckit import randint_choice
from tqdm import tqdm

class SGL(nn.Module):
    def __init__(self, data_config, args):
        super(SGL, self).__init__()
        print('use SGL')
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
        self.n_layers = args.n_layers #2
        self.temp = args.temp
        self.batch_size = args.batch_size
        self.emb_reg = args.emb_reg #2e-5
        self.ssl_reg = args.ssl_reg #0.1
        self.zu,self.zuu,self.zi,self.zii=None,None,None,None
        self.hu,self.hi,self.eu,self.ei=None,None,None,None
        #self.tail_neg=self.neg_sample(self.all_h_list,self.all_t_list)
        self.nhu,self.nhi=None,None
        self.zero=torch.tensor(0).cuda()
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim) #(u,d)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim) #(I,d)
        self.dropout=args.dropout
        self.G_values1,self.G_values2=self.ED(self.G_values,self.dropout),self.ED(self.G_values,self.dropout)
        self._init_weight()

    def ED(self,tensor, percent=0.25):
        num_elements = tensor.numel() 
        num_zeros = int(num_elements * percent)  
        indices = torch.randperm(num_elements)[:num_zeros] 
        tensor[indices] = 0 
        return tensor

    def _init_weight(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def inference1(self):
        all_embeddings = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        all_embeddings1 = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        all_embeddings2 = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        for i in range(0, self.n_layers):
            g=torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], all_embeddings[i])
            g1= torch_sparse.spmm(self.G_indices, self.G_values1, self.A_in_shape[0], self.A_in_shape[1], all_embeddings1[i])
            g2 = torch_sparse.spmm(self.G_indices, self.G_values2, self.A_in_shape[0], self.A_in_shape[1],all_embeddings2[i])
            all_embeddings.append(g)
            all_embeddings1.append(g1)
            all_embeddings2.append(g2)
        #[54335, 3, 32]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings1 = torch.stack(all_embeddings1, dim=1)
        all_embeddings2 = torch.stack(all_embeddings2, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        all_embeddings1 = torch.sum(all_embeddings1, dim=1, keepdim=False)
        all_embeddings2 = torch.sum(all_embeddings2, dim=1, keepdim=False)
        self.ua_embedding, self.ia_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        self.zu,self.zi=torch.split(all_embeddings1, [self.n_users, self.n_items], 0)
        self.zuu, self.zii = torch.split(all_embeddings2, [self.n_users, self.n_items], 0)



    def sgl_loss(self, users, items, zu,zuu,zi,zii,t):
        users = torch.unique(users)
        items = torch.unique(items)
        cl_loss = 0.0
        u1 = F.normalize(zu[users], dim=1)
        i1 = F.normalize(zi[items], dim=1)
        u2 = F.normalize(zuu[users], dim=1)
        i2 = F.normalize(zii[items], dim=1)
        cl_loss += self.cal_loss(u1, u2,t) 
        cl_loss += self.cal_loss(i1, i2,t) 
        return cl_loss

    def globalcl(self, users, items, eu,hu,ei,hi,t):
        users = torch.unique(users)
        items = torch.unique(items)
        cl_loss = 0.0
        eu = F.normalize(eu[users], dim=1)
        ei = F.normalize(ei[items], dim=1)
        hu = F.normalize(hu[users], dim=1)
        hi = F.normalize(hi[items], dim=1)
        cl_loss += self.cal_loss(eu,hu,t) 
        cl_loss += self.cal_loss(ei,hi,t)
        return cl_loss


    def bpr_loss(self,user,pos,neg):
        pos_scores = torch.sum(user* pos, 1)
        neg_scores = torch.sum(user * neg, 1)
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return bpr_loss

    def forward(self, users, pos_items, neg_items):
        users = torch.LongTensor(users).cuda()
        pos_items = torch.LongTensor(pos_items).cuda()
        neg_items = torch.LongTensor(neg_items).cuda()
        self.inference1()
        # bpr
        #gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings
        #(1024,32)
        u_emb = self.ua_embedding[users]
        pos_emb = self.ia_embedding[pos_items] # (1024,32)
        neg_emb = self.ia_embedding[neg_items] # (1024,32)
        # hu_emb = self.hu[users]
        # hpos_emb = self.hi[pos_items]  # (1024,32)
        # hneg_emb = self.hi[neg_items]  # (1024,32)

        u_embeddings_pre = self.user_embedding(users)
        pos_embeddings_pre = self.item_embedding(pos_items)
        neg_embeddings_pre = self.item_embedding(neg_items)
        emb_loss = (u_embeddings_pre.norm(2).pow(2) + pos_embeddings_pre.norm(2).pow(2) + neg_embeddings_pre.norm(2).pow(2))
        emb_loss = self.emb_reg * emb_loss

        # self-supervise learning
        bpr_loss = self.bpr_loss(u_emb, pos_emb, neg_emb)
        cse_loss = self.ssl_reg * self.sgl_loss(users, pos_items,self.zu,self.zuu,self.zi,self.zii,self.temp)
        svd_loss=self.zero
        #svd_loss = self.ssl_reg * self.cse_loss(users, pos_items, self.zu3, self.zuu3, self.zi3, self.zii3, self.temp)
        #cse_loss,svd_loss=self.zero,self.zero
        #svd_loss+= self.ssl_reg * self.globalcl(users, pos_items,self.ua_embedding,self.nhu,self.ia_embedding,self.nhi,self.temp)
        return bpr_loss,  svd_loss, cse_loss,emb_loss

    def predict(self, users):
        u_embeddings = self.ua_embedding[torch.LongTensor(users).cuda()]
        i_embeddings = self.ia_embedding
        batch_ratings = torch.matmul(u_embeddings, i_embeddings.T)
        return batch_ratings
