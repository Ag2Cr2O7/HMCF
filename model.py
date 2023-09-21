import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from reckit import randint_choice
from tqdm import tqdm

class DCCF(nn.Module):
    def __init__(self, data_config, args):
        super(DCCF, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.plain_adj = data_config['plain_adj']
        self.all_h_list = data_config['all_h_list'] #[交互用户|交互项目[i]+29601]
        self.all_t_list = data_config['all_t_list'] #[29601+交互项目[i]|交互用户]
        self.A_in_shape = self.plain_adj.tocoo().shape # (u+i,u+i)
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).cuda() #将【row和col】一起放入gpu
        self.D_indices = torch.tensor([list(range(self.n_users + self.n_items)), list(range(self.n_users + self.n_items))], dtype=torch.long).cuda() #[2, 54335]，元素为0~u+i
        self.all_h_list = torch.LongTensor(self.all_h_list).cuda()
        self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
        #归一化矩阵的索引和值
        self.G_indices, self.G_values = self._cal_sparse_adj()

        self.emb_dim = args.embed_size
        self.n_layers = args.n_layers #2
        self.n_intents = args.n_intents #128
        self.temp = args.temp


        self.batch_size = args.batch_size
        self.emb_reg = args.emb_reg #2e-5
        self.cen_reg = args.cen_reg #0.005
        self.ssl_reg = args.ssl_reg #0.1

        # self.lsvd = args.lsvd
        # self.tsvd = args.tsvd

        self.dropout=args.dropout
        self.zu,self.zuu,self.zi,self.zii=None,None,None,None
        self.zu3,self.zuu3,self.zi3,self.zii3=[None] * self.n_layers,[None] * self.n_layers,[None] * self.n_layers,[None] * self.n_layers
        self.act = nn.LeakyReLU(0.8)
        self.hu,self.hi,self.eu,self.ei=None,None,None,None
        #self.tail_neg=self.neg_sample(self.all_h_list,self.all_t_list)
        self.nhu,self.nhi=None,None
        self.zero=torch.tensor(0).cuda()

        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim) #(u,d)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim) #(I,d)

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def neg_sample(self,users,items):
        user_set={}
        for i,user in enumerate(tqdm(users)):
            if user not in user_set:
                user_set[user]=[items[i]]
            else:
                user_set[user].append(items[i])
        neg=[]
        for user in tqdm(user_set):
            while True:
                #随机选择一个项目
                temp=np.random.randint(0, self.n_items-1)
                #如果用户交互过了，就重新选，直到用户没交互过
                if temp in user_set[user]:
                    continue
                else:
                    neg.append(temp)
                    break
        return torch.tensor(neg).cuda()




    def _cal_sparse_adj(self):
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda() #全1数组
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape).cuda() #(54335, 54335)
        D_values = A_tensor.sum(dim=1).pow(-0.5)
        self.A_in_shape = self.plain_adj.tocoo().shape  # (u+i,u+i)
        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        # 归一化矩阵的索引和值
        return G_indices, G_values

    def perturb_embedding(self, embeds,eps=0.2):
        # torch.sign将emb中所有的符号提取出来
        noise = (F.normalize(torch.rand(embeds.shape).cuda(), p=2) * torch.sign(embeds)) * eps
        return embeds + noise

    def inference1(self):
        all_embeddings = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        all_embeddings1 = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        all_embeddings2 = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        for i in range(0, self.n_layers):
            g=torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], all_embeddings[i])
            g1= torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], all_embeddings1[-1])
            g2 = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1],all_embeddings2[-1])
            zqg1 = self.perturb_embedding(g1,eps=0.2)  # eps控制噪声的大小
            zqg2 = self.perturb_embedding(g2, eps=0.2)
            all_embeddings.append(g)
            all_embeddings1.append(zqg1)
            all_embeddings2.append(zqg2)
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


    def cal_loss(self,emb1, emb2,t):
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / t) # (8686,32)*(8686,32)->(8686)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / t), axis=1) # (8686,32)@(32,8686)=(8686,8686)->(8686)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]
        return loss

    def cse_loss(self, users, items, zu,zuu,zi,zii,t):
        users = torch.unique(users)
        items = torch.unique(items)
        cl_loss = 0.0
        for i in range(self.n_layers):
            u1 = F.normalize(zu[i][users], dim=1)
            i1 = F.normalize(zi[i][items], dim=1)
            u2 = F.normalize(zuu[i][users], dim=1)
            i2 = F.normalize(zii[i][items], dim=1)
            cl_loss += self.cal_loss(u1, u2,t) #用户的gnn和意图对比
            cl_loss += self.cal_loss(i1, i2,t) #用户的gnn和gnn掩码对比
        return cl_loss

    def sgl_loss(self, users, items, zu,zuu,zi,zii,t):
        users = torch.unique(users)
        items = torch.unique(items)
        cl_loss = 0.0
        u1 = F.normalize(zu[users], dim=1)
        i1 = F.normalize(zi[items], dim=1)
        u2 = F.normalize(zuu[users], dim=1)
        i2 = F.normalize(zii[items], dim=1)
        cl_loss += self.cal_loss(u1, u2,t) #用户的gnn和意图对比
        cl_loss += self.cal_loss(i1, i2,t) #用户的gnn和gnn掩码对比
        return cl_loss

    def globalcl(self, users, items, eu,hu,ei,hi,t):
        users = torch.unique(users)
        items = torch.unique(items)
        cl_loss = 0.0
        eu = F.normalize(eu[users], dim=1)
        ei = F.normalize(ei[items], dim=1)
        hu = F.normalize(hu[users], dim=1)
        hi = F.normalize(hi[items], dim=1)
        cl_loss += self.cal_loss(eu,hu,t) #用户的gnn和意图对比
        cl_loss += self.cal_loss(ei,hi,t) #用户的gnn和gnn掩码对比
        return cl_loss


    def cal_ssl_loss(self, users, items, gnn_emb, int_emb, gaa_emb, iaa_emb):
        users = torch.unique(users)
        items = torch.unique(items)
        cl_loss = 0.0

        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.temp)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.temp), axis=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss

        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.n_users, self.n_items], 0)
            u_int_embs, i_int_embs = torch.split(int_emb[i], [self.n_users, self.n_items], 0)
            u_gaa_embs, i_gaa_embs = torch.split(gaa_emb[i], [self.n_users, self.n_items], 0)
            u_iaa_embs, i_iaa_embs = torch.split(iaa_emb[i], [self.n_users, self.n_items], 0)

            u_gnn_embs = F.normalize(u_gnn_embs[users], dim=1)
            u_int_embs = F.normalize(u_int_embs[users], dim=1)
            u_gaa_embs = F.normalize(u_gaa_embs[users], dim=1)
            u_iaa_embs = F.normalize(u_iaa_embs[users], dim=1)

            i_gnn_embs = F.normalize(i_gnn_embs[items], dim=1)
            i_int_embs = F.normalize(i_int_embs[items], dim=1)
            i_gaa_embs = F.normalize(i_gaa_embs[items], dim=1)
            i_iaa_embs = F.normalize(i_iaa_embs[items], dim=1)

            cl_loss += cal_loss(u_gnn_embs, u_int_embs) #用户的gnn和意图对比
            cl_loss += cal_loss(u_gnn_embs, u_gaa_embs) #用户的gnn和gnn掩码对比
            cl_loss += cal_loss(u_gnn_embs, u_iaa_embs) #用户的gnn和意图掩码对比

            cl_loss += cal_loss(i_gnn_embs, i_int_embs)
            cl_loss += cal_loss(i_gnn_embs, i_gaa_embs)
            cl_loss += cal_loss(i_gnn_embs, i_iaa_embs)

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
        # 图和意图，掩码图和掩码意图
        #gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings
        #(1024,32)
        u_emb = self.ua_embedding[users]
        pos_emb = self.ia_embedding[pos_items] # (1024,32)
        neg_emb = self.ia_embedding[neg_items] # (1024,32)
        # hu_emb = self.hu[users]
        # hpos_emb = self.hi[pos_items]  # (1024,32)
        # hneg_emb = self.hi[neg_items]  # (1024,32)

        # l2正则化
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