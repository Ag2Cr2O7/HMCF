import torch.optim as optim
import random
import logging
import datetime
import os
from utility.batch_test import *
from utility.load_data import *
from model import *
from tqdm import tqdm
from time import time
from copy import deepcopy
from datetime import datetime
import importlib

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

if __name__ == '__main__':
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    mydata = 'gowalla'
    mymodel= 'hmcf'
    parser_name = f"myparser.{mymodel+'parser'}.py"
    f='parse_args'
    pmodule = importlib.import_module(parser_name.replace('.py', ''))
    fargs = getattr(pmodule, 'parse_args')
    args = fargs(dataset=mydata)
    uselog=True
    set_seed(args.seed)
    curr_time = datetime.now()
    print(curr_time)
    filename = mydata + mymodel
    if not os.path.exists('log'):
        os.mkdir('log')
    if uselog:
        logger = logging.getLogger('train_logger')
        logger.setLevel(logging.INFO)
        print(filename)
        logfile = logging.FileHandler('log/{}.log'.format(filename), 'a', encoding='utf-8')
        logfile.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        logfile.setFormatter(formatter)
        logger.addHandler(logfile)
    data_generator = Data(args)
    if uselog:logger.info(data_generator.get_statistics())
    print("************************* Run with following settings 🏃 ***************************")
    print(args)
    if uselog:logger.info(args)
    print("************************************************************************************")
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    plain_adj = data_generator.get_adj_mat()
    all_h_list, all_t_list, all_v_list = load_adjacency_list_data(plain_adj)
    config['plain_adj'] = plain_adj
    config['all_h_list'] = all_h_list
    config['all_t_list'] = all_t_list
    module_name = f"baseline.{mymodel}.py"
    f_name = mymodel
    module = importlib.import_module(module_name.replace('.py', ''))
    # if hasattr(module, f_name):
    baseline_model = getattr(module, f_name)
    _model = baseline_model(config, args).cuda()
    optimizer = optim.Adam(_model.parameters(), lr=args.lr)
    # else:
    #     print(f"Function '{function_name}' not found in module '{module_name}'")
    print("Start Training")
    stopping_step = 0
    last_state_dict = None
    R20, N20, R40, N40 = 0, 0, 0, 0
    Epoch = 0
    for epoch in range(args.epoch):
        ## train
        t1 = time()
        n_samples = data_generator.uniform_sample() #mini_batch*batch_size
        n_batch = int(np.ceil(n_samples / args.batch_size)) #mini_batch
        _model.train()
        loss, mf_loss, emb_loss, cen_loss, cl_loss = 0., 0., 0., 0., 0.
        encoder, decoder = None, None
        sampScores=0
        for idx in tqdm(range(n_batch)):
            optimizer.zero_grad()
            users, pos_items, neg_items = data_generator.mini_batch(idx)
            batch_mf_loss, batch_emb_loss, batch_cen_loss, batch_cl_loss = _model(users, pos_items, neg_items)
            batch_loss = batch_mf_loss + batch_emb_loss + batch_cen_loss + batch_cl_loss
            loss += float(batch_loss) / n_batch
            mf_loss += float(batch_mf_loss) / n_batch
            emb_loss += float(batch_emb_loss) / n_batch
            cen_loss += float(batch_cen_loss) / n_batch
            cl_loss += float(batch_cl_loss) / n_batch
            batch_loss.backward()
            optimizer.step()

        last_state_dict = deepcopy(_model.state_dict())
        torch.cuda.empty_cache()

        if epoch % args.show_step != 0 and epoch != args.epoch - 1:
            perf_str = 'Epoch %2d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f]' % (epoch, time() - t1, loss, mf_loss, emb_loss, cen_loss, cl_loss)
            print(perf_str)
            if uselog:logger.info(perf_str)
            continue

        t2 = time()

        with torch.no_grad():
            _model.eval()
            _model.message_pass()
            test_ret = eval_PyTorch(_model, data_generator, eval(args.Ks))
            torch.cuda.empty_cache()
        t3 = time()
        perf_str = 'Epoch %2d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f]\nTest-Recall=[%.4f, %.4f], Test-NDCG =[%.4f, %.4f]' % \
                   (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, cen_loss, cl_loss, test_ret['recall'][0], test_ret['recall'][1], test_ret['ndcg'][0], test_ret['ndcg'][1])
        print(perf_str)
        if uselog:logger.info(perf_str)
        if test_ret['recall'][0]>R20 and test_ret['ndcg'][0]>N20:
            R20,R40,N20,N40=test_ret['recall'][0], test_ret['recall'][1], test_ret['ndcg'][0], test_ret['ndcg'][1]
            Epoch=epoch

    if not os.path.exists('saved'):
        os.mkdir('saved')
    if args.save_model:
        torch.save(last_state_dict, 'saved/{}.pth'.format(args.dataset))
    _model.load_state_dict(last_state_dict)
    with torch.no_grad():
        _model.eval()
        _model.inference1()
        final_test_ret = eval_PyTorch(_model, data_generator, eval(args.Ks))

    pref_str = 'Final Test Set Result: Test-Recall=[%.4f, %.4f], Test-NDCG=[%.4f, %.4f]' % (final_test_ret['recall'][0], final_test_ret['recall'][1], final_test_ret['ndcg'][0], final_test_ret['ndcg'][1])
    print(pref_str)
    if uselog:logger.info(pref_str)
    print('log/{}.log'.format(filename))
    print('---------------Best epoch: {}-------------------'.format(Epoch))
    print('Recall@20: {} NDCG@20: {} Recall@40: {} NDCG@40: {}'.format(R20,N20,R40,N40))
