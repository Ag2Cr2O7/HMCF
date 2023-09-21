import torch
import numpy as np
from tqdm import trange,tqdm
import random
import pickle

#r=[array([1., 0., 1., 1.])]表示预测数组的第二个元素是没有预测成功的
def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        # 将predictTopK中的每个元素在groundTrue中的存在性映射成True或False的列表
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float") #将映射结果转换成np.array格式，并将其元素的数据类型转换为float
        r.append(pred) #将转换后的标签数据pred添加到列表r中。
    return np.array(r).astype('float')

def getLabel_spuser(test_data, pred_data,low=0,high=10):
    r = []
    #for i in range(len(test_data)):
    for i in range(low,high):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        # 将predictTopK中的每个元素在groundTrue中的存在性映射成True或False的列表
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float") #将映射结果转换成np.array格式，并将其元素的数据类型转换为float
        r.append(pred) #将转换后的标签数据pred添加到列表r中。
    return np.array(r).astype('float')

def getLabel_item(test_data, pred_data,item_set):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        def _check_condition(x):
                return x in groundTrue and x in item_set
        pred = list(map(_check_condition, predictTopK))
        #pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float") #将映射结果转换成np.array格式，并将其元素的数据类型转换为float
        r.append(pred) #将转换后的标签数据pred添加到列表r中。
    return np.array(r).astype('float')



def Recall_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    return recall


def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1] # 10240的list，记录了测试集
    r = getLabel(groundTrue, sorted_items) #预测的0,1矩阵
    recall, ndcg = [], []
    for k in topks:
        recall.append(Recall_ATk(groundTrue, r, k))
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'ndcg': np.array(ndcg)}



def test_one_batch_item(X, topks,item_set):
    sorted_items = X[0].numpy()
    groundTrue = X[1] # 10240的list，记录了测试集
    r = getLabel_item(groundTrue, sorted_items,item_set) #预测的0,1矩阵
    recall, ndcg = [], []
    for k in topks:
        recall.append(Recall_ATk(groundTrue, r, k))
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'ndcg': np.array(ndcg)}

def test_user(X, topks,userg):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel_spuser(groundTrue, sorted_items,userg) #预测的0,1矩阵
    recall, ndcg = [], []
    for k in topks:
        recall.append(Recall_ATk(groundTrue, r, k))
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'ndcg': np.array(ndcg)}

def eval_PyTorch_user(model, data_generator, Ks,sparse_user):
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    test_users_pre = list(sparse_user) #47528
    # 把测试集中没有的去了
    test_users=[]
    for i in range(len(test_users_pre)):
        if test_users_pre[i] in data_generator.test_set:
            test_users.append(test_users_pre[i])

    u_batch_size = data_generator.batch_size

    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    batch_rating_list = []
    ground_truth_list = []
    count = 0
    for u_batch_id in trange(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        rate_batch = model.predict(user_batch)

        count += rate_batch.shape[0]

        exclude_index = []
        exclude_items = []
        ground_truth = []
        for i in range(len(user_batch)):
            if user_batch[i] in data_generator.train_items: #新加
                train_items = list(data_generator.train_items[user_batch[i]])
            exclude_index.extend([i] * len(train_items))
            exclude_items.extend(train_items)
            ground_truth.append(list(data_generator.test_set[user_batch[i]]))
        rate_batch[exclude_index, exclude_items] = -(1 << 20)
        _, rate_batch_k = torch.topk(rate_batch, k=max(Ks))
        batch_rating_list.append(rate_batch_k.cpu())
        ground_truth_list.append(ground_truth)

    X = zip(batch_rating_list, ground_truth_list)
    batch_results = []
    for x in X:
        batch_results.append(test_one_batch(x, Ks))
    for batch_result in batch_results:
        result['recall'] += batch_result['recall'] / n_test_users
        result['ndcg'] += batch_result['ndcg'] / n_test_users

    assert count == n_test_users

    return result

def eval_PyTorch(model, data_generator, Ks):
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    test_users = list(data_generator.test_set.keys()) #47528

    u_batch_size = data_generator.batch_size

    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    batch_rating_list = []
    ground_truth_list = []
    count = 0
    for u_batch_id in trange(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        rate_batch = model.predict(user_batch)

        count += rate_batch.shape[0]

        exclude_index = []
        exclude_items = []
        ground_truth = []
        for i in range(len(user_batch)):
            if user_batch[i] in data_generator.train_items: #新加
                train_items = list(data_generator.train_items[user_batch[i]])
            exclude_index.extend([i] * len(train_items))
            exclude_items.extend(train_items)
            ground_truth.append(list(data_generator.test_set[user_batch[i]]))
        rate_batch[exclude_index, exclude_items] = -(1 << 20)
        _, rate_batch_k = torch.topk(rate_batch, k=max(Ks))
        batch_rating_list.append(rate_batch_k.cpu())
        ground_truth_list.append(ground_truth)

    X = zip(batch_rating_list, ground_truth_list)
    batch_results = []
    for x in X:
        batch_results.append(test_one_batch(x, Ks))
    for batch_result in batch_results:
        result['recall'] += batch_result['recall'] / n_test_users
        result['ndcg'] += batch_result['ndcg'] / n_test_users

    assert count == n_test_users

    return result


def eval_PyTorch_item(model, data_generator, Ks,item_set):
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    test_users = list(data_generator.test_set.keys()) #47528

    u_batch_size = data_generator.batch_size

    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    batch_rating_list = []
    ground_truth_list = []
    count = 0
    for u_batch_id in trange(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        rate_batch = model.predict(user_batch)

        count += rate_batch.shape[0]

        exclude_index = []
        exclude_items = []
        ground_truth = []
        for i in range(len(user_batch)):
            if user_batch[i] in data_generator.train_items: #新加
                train_items = list(data_generator.train_items[user_batch[i]])
            exclude_index.extend([i] * len(train_items))
            exclude_items.extend(train_items)
            ground_truth.append(list(data_generator.test_set[user_batch[i]]))
        rate_batch[exclude_index, exclude_items] = -(1 << 20)
        _, rate_batch_k = torch.topk(rate_batch, k=max(Ks))
        batch_rating_list.append(rate_batch_k.cpu())
        ground_truth_list.append(ground_truth)

    X = zip(batch_rating_list, ground_truth_list)
    batch_results = []
    for x in X:
        batch_results.append(test_one_batch_item(x, Ks,item_set))
    for batch_result in batch_results:
        result['recall'] += batch_result['recall'] / n_test_users
        result['ndcg'] += batch_result['ndcg'] / n_test_users

    assert count == n_test_users

    return result

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_adjacency_list_data(adj_mat):
    tmp = adj_mat.tocoo()
    all_h_list = list(tmp.row)
    all_t_list = list(tmp.col)
    all_v_list = list(tmp.data)

    return all_h_list, all_t_list, all_v_list

def user_sparse(path,low=0,high=15):
    device = 'cuda:0'  if torch.cuda.is_available() else 'cpu'
    print('load:'+path + 'trnMat.pkl')
    f1 = open(path + 'trnMat.pkl', 'rb')
    train = pickle.load(f1)  # (u1,i5)->1.0 (u1,i8)->1.0
    a = train.row
    b = train.col
    train_dict={}
    user_dict={}
    for i in tqdm(range(len(a))):
        if a[i] not in train_dict:
            train_dict[a[i]]=[]
            train_dict[a[i]].append(b[i])
        else:
            train_dict[a[i]].append(b[i])
    for user in train_dict:
        if low <= len(train_dict[user]) < high:
            user_dict[user]=train_dict[user]
    sparse_user=[key for key in user_dict]
    sparse_user.sort()
    print('{}-{}用户数：{}'.format(low,high,len(sparse_user)))
    return np.array(sparse_user)

def item_popular(path,low=0,high=15):
    f1 = open(path + 'trnMat.pkl', 'rb')
    train = pickle.load(f1)  # (u1,i5)->1.0 (u1,i8)->1.0
    b = train.col
    item_dict={}
    popu_item=[]
    for i in tqdm(range(len(b))):
        if b[i] not in item_dict:
            item_dict[b[i]]=1
        else:
            item_dict[b[i]]+=1
    #print(item_dict)
    for item in item_dict:
        if low <= item_dict[item] < high:
            popu_item.append(item)
    popu_item.sort()
    print('{}-{}项目数：{}'.format(low,high,len(popu_item)))
    return np.array(popu_item)