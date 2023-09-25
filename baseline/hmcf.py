import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from reckit import randint_choice
from manifold.hyperboloid import Hyperboloid


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
        self.D_indices = torch.tensor([list(range(self.n_users + self.n_items)), list(range(self.n_users + self.n_items))], dtype=torch.long).cuda() #[2, 54335]，元素为0~u+i
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
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        self.suser_embedding=nn.Embedding(self.n_users, self.emb_dim+1)
        self.sitem_embedding=nn.Embedding(self.n_items, self.emb_dim+1)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        nn.init.xavier_normal_(self.suser_embedding.weight)
        nn.init.xavier_normal_(self.sitem_embedding.weight)
        self.suser_embedding.weight.data[:, 0] = 0.0
        self.sitem_embedding.weight.data[:, 0] = 0.0

    def _cal_sparse_adj(self):
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)
        self.A_in_shape = self.plain_adj.tocoo().shape
        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        return G_indices, G_values

    def message_pass(self):
        all_embeddings = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        sall_embeddings = [torch.concat([self.suser_embedding.weight, self.sitem_embedding.weight], dim=0)]
        for i in range(0, self.n_layers):
            zqg1=nn.functional.dropout(all_embeddings[i], p=0)
            zqg2=nn.functional.dropout(all_embeddings[i], p=0.1)
            gnn_emb1= torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], zqg1)
            gnn_emb2 = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1],zqg2)
            self.zu[i], self.zi[i] = torch.split(gnn_emb1, [self.n_users, self.n_items], 0)
            self.zuu[i], self.zii[i] = torch.split(gnn_emb2, [self.n_users, self.n_items], 0)
            szq=nn.functional.dropout(sall_embeddings[i], p=0.1)
            G_hyper=torch_sparse.spmm(self.G_indices,self.G_values, self.A_in_shape[0], self.A_in_shape[1], szq)
            self.hu[i], self.hi[i] = torch.split(G_hyper, [self.n_users, self.n_items], 0)
            hnn = torch.concat([self.hu[i], self.hi[i]], dim=0)
            gnn1 = torch.concat([self.zu[i], self.zi[i]], dim=0)
            all_embeddings.append(gnn1+all_embeddings[i])
            sall_embeddings.append( hnn+ sall_embeddings[i])
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        self.ua_embedding, self.ia_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], 0)


    def cal_loss(self,emb1, emb2,t):
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / t)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / t), axis=1)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]
        return loss

    def cse_loss(self, users, positems, zu,zuu,zi,zii,t):
        users = torch.unique(users)
        items = torch.unique(positems)
        cl_loss = 0.0
        for i in range(self.n_layers):
            u1 = F.normalize(zu[i][users], dim=1)
            i1 = F.normalize(zi[i][items], dim=1)
            u2 = F.normalize(zuu[i][users], dim=1)
            i2 = F.normalize(zii[i][items], dim=1)
            cl_loss += self.cal_loss(u1, u2,t)
            cl_loss += self.cal_loss(i1, i2,t)
        return cl_loss


    def cross_cl(self, users, positems, zu, hu, zi, hi, t):
        users = torch.unique(users)
        items = torch.unique(positems)
        zerou = torch.zeros(users.shape[0], 1).cuda()
        zeroi = torch.zeros(items.shape[0], 1).cuda()
        cl_loss = 0.0
        for i in range(self.n_layers):
            u1 = torch.cat([zerou, zu[i][users]], dim=1).cuda()
            i1 = torch.cat([zeroi, zi[i][items]], dim=1).cuda()
            u1 = F.normalize(self.manifold.expmap0(u1,1), dim=1)
            i1 = F.normalize(self.manifold.expmap0(i1,1), dim=1)
            u2 = F.normalize(hu[i][users], dim=1)
            i2 = F.normalize(hi[i][items], dim=1)
            pos_score = torch.sum(torch.exp(-self.manifold.sqdist(u1,u2,c=1) / t), axis=1)
            negu=torch.flip(u2, dims=[0])
            neg_score = self.hweight*torch.sum(torch.exp(-self.manifold.sqdist(u1,negu,c=1) / t), axis=1)
            unan=-torch.log(pos_score / (neg_score + 1e-8) + 1e-8)
            unan[torch.isnan(unan)] = 0
            unan[torch.isinf(unan)]=0
            lossu = torch.sum(unan)
            pos_score = torch.sum(torch.exp(-self.manifold.sqdist(i1, i2, c=1)/t), axis=1)
            negi = torch.flip(i2, dims=[0])
            neg_score =self.hweight* torch.sum(torch.exp(-self.manifold.sqdist(i1, negi, c=1) / t), axis=1)
            inan = -torch.log(pos_score / (neg_score + 1e-8) + 1e-8)
            inan[torch.isnan(inan)] = 0
            inan[torch.isinf(inan)]=0
            lossi = torch.sum(inan)
            cl_loss += lossu
            cl_loss += lossi
            cl_loss /= pos_score.shape[0]
        return cl_loss

    def emb_loss(self,u_embeddings_pre,pos_embeddings_pre,neg_embeddings_pre):
        emb_loss = (u_embeddings_pre.norm(2).pow(2) + pos_embeddings_pre.norm(2).pow(2) + neg_embeddings_pre.norm(2).pow(2))
        emb_loss = self.emb_reg * emb_loss
        return emb_loss


    def bpr_loss(self,u_embeddings,pos_embeddings,neg_embeddings):
        pos_scores = torch.sum(u_embeddings * pos_embeddings, 1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, 1)
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return bpr_loss


    def forward(self, users, pos_items, neg_items):
        users = torch.LongTensor(users).cuda()
        pos_items = torch.LongTensor(pos_items).cuda()
        neg_items = torch.LongTensor(neg_items).cuda()
        self.message_pass()
        u_embeddings = self.ua_embedding[users]
        pos_embeddings = self.ia_embedding[pos_items]
        neg_embeddings = self.ia_embedding[neg_items]
        mf_loss=self.bpr_loss(u_embeddings,pos_embeddings,neg_embeddings)
        u_embeddings_pre = self.user_embedding(users)
        pos_embeddings_pre = self.item_embedding(pos_items)
        neg_embeddings_pre = self.item_embedding(neg_items)
        emb_loss=self.emb_loss(u_embeddings_pre,pos_embeddings_pre,neg_embeddings_pre)
        loss_s = self.ssl_reg * self.cse_loss(users, pos_items,self.zu,self.zuu,self.zi,self.zii,self.temp)
        loss_h = self.lsvd * self.cross_cl(users, pos_items, self.zu, self.hu, self.zi, self.hi, self.htemp)
        return mf_loss,  loss_h, loss_s,emb_loss

    def predict(self, users):
        u_embeddings = self.ua_embedding[torch.LongTensor(users).cuda()]
        i_embeddings = self.ia_embedding
        batch_ratings = torch.matmul(u_embeddings, i_embeddings.T)
        return batch_ratings
