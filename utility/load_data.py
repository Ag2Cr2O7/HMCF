import pickle
import numpy as np
from time import time
from tqdm import tqdm
import scipy.sparse as sp

class Data(object):
    def __init__(self, args):

        self.path = args.data_path + args.dataset #'data/yelp'
        self.n_batch = args.n_batch
        self.batch_size = args.batch_size
        self.train_num = args.train_num
        self.sample_num = args.sample_num

        train_file = self.path + '/trnMat.pkl'
        test_file = self.path + '/tstMat.pkl'
        with open(train_file, 'rb') as f:
            train_mat = pickle.load(f)
        with open(test_file, 'rb') as f:
            test_mat = pickle.load(f)

        # get number of users and items
        self.n_users, self.n_items = train_mat.shape[0], train_mat.shape[1]
        self.n_train, self.n_test = len(train_mat.row), len(test_mat.row)

        self.print_statistics()

        self.R = train_mat.todok() #稀疏矩阵转密集字典表示，节约空间
        self.train_items, self.test_set = {}, {}
        train_uid, train_iid = train_mat.row, train_mat.col
        for i in range(len(train_uid)):
            uid = train_uid[i]
            iid = train_iid[i]
            if uid not in self.train_items:
                self.train_items[uid] = [iid]
            else:
                self.train_items[uid].append(iid) #训练集用户交互的项目保存为字典形式
        test_uid, test_iid = test_mat.row, test_mat.col
        for i in range(len(test_uid)):
            uid = test_uid[i]
            iid = test_iid[i]
            if uid not in self.test_set:
                self.test_set[uid] = [iid]
            else:
                self.test_set[uid].append(iid) #测试集用户交互的项目保存为字典形式

    def get_adj_mat(self):
        adj_mat = self.create_adj_mat()
        return adj_mat

    def create_adj_mat(self):
        t1 = time()
        rows = self.R.tocoo().row #交互的用户1069128 #
        cols = self.R.tocoo().col #交互的项目1069128
        new_rows = np.concatenate([rows, cols + self.n_users], axis=0) # [用户|用户[i]+项目[i]],len=2138256
        new_cols = np.concatenate([cols + self.n_users, rows], axis=0) # [用户[i]+项目[i]|项目]
        #row=array([    0,     0,     0, ..., 54334, 54334, 54334], dtype=int32)
        #col=array([29601, 29602, 29603, ..., 26375, 26574, 29310], dtype=int32)
        #data=array([1., 1., 1., ..., 1., 1., 1.])
        adj_mat = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=[self.n_users + self.n_items, self.n_users + self.n_items]).tocsr().tocoo() #(54335, 54335)(user+item,user_item)
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        return adj_mat.tocsr() #压缩存储

    def uniform_sample(self):
        #从0到self.n_users-1之间随机选择n_batch*batch_size个用户，构成一个用户集合users
        # n_batch*batch_size>self.n_user会重复取！
        users = np.random.randint(0, self.n_users, int(self.n_batch * self.batch_size)) #40*10240=40960
        train_data = []
        for i, user in tqdm(enumerate(users), desc='Sampling Data', total=len(users)):
            # train_items为字典，读取训练集每个用户交互的项目
            if user in self.train_items:
                pos_for_user = self.train_items[user]
            # 随机选择该用户的一个交互项目作为正例
            pos_index = np.random.randint(0, len(pos_for_user))
            pos_item = pos_for_user[pos_index]
            while True:
                #随机选择一个项目
                neg_item = np.random.randint(0, self.n_items)
                #如果用户交互过了，就重新选，直到用户没交互过
                if self.R[user, neg_item] == 1:
                    continue
                else:
                    break
            #[[用户1, 一个pos, 一个neg],[用户2, 一个pos, 一个neg]]
            train_data.append([user, pos_item, neg_item])
        self.train_data = np.array(train_data)
        return len(self.train_data)

    def mini_batch(self, batch_idx):
        st = batch_idx * self.batch_size #0
        ed = min((batch_idx + 1) * self.batch_size, len(self.train_data)) #1024
        batch_data = self.train_data[st: ed] #取前batchsize个
        users = batch_data[:, 0] #batch_size个用户,注意里面的userid会有重复
        pos_items = batch_data[:, 1] #batchsize=1024
        neg_items = batch_data[:, 2] #batch
        #batch_size个用户(有重复)，对应的pos和neg
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train)/(self.n_users * self.n_items)))

    def get_statistics(self):
        sta = ""
        sta += 'n_users=%d, n_items=%d\t' % (self.n_users, self.n_items)
        sta += 'n_interactions=%d\t' % (self.n_train + self.n_test)
        sta += 'n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train)/(self.n_users * self.n_items))
        return sta
