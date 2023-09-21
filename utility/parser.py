import argparse
# DCCF gowalla[0.1876 0.2644 0.1123 0.1323] amazon[0.0889 0.1343 0.0680 0.0829] tmall[0.0668 0.1042 0.0469 0.0598]
def parse_args(dataset):
    if dataset=='yelp':
        parser = argparse.ArgumentParser(description="Run MDGCL.")
        parser.add_argument('--data_path', nargs='?', default='data/', help='Input data path.')
        parser.add_argument('--seed', type=int, default=1024, help='random seed')
        parser.add_argument('--dataset', nargs='?', default=dataset,help='Choose a dataset from {gowalla, amazon, tmall}')
        parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
        parser.add_argument('--save_model', type=bool, default=False, help='Whether to save')
        parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
        parser.add_argument('--embed_size', type=int, default=32, help='Embedding size.')
        parser.add_argument('--n_batch', type=int, default=40, help='Number of mini-batches')
        parser.add_argument('--batch_size', type=int, default=10240, help='batch size')
        parser.add_argument('--train_num', type=int, default=10000, help='Number of training instances per epoch')
        parser.add_argument('--sample_num', type=int, default=40, help='Number of pos/neg samples for each instance')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
        parser.add_argument('--emb_reg', type=float, default=1e-4, help='Regularizations.')
        parser.add_argument('--cen_reg', type=float, default=1e-4, help='Regularizations.')
        parser.add_argument('--ssl_reg', type=float, default=1e-0, help='Reg weight for ssl loss')
        parser.add_argument('--n_layers', type=int, default=2, help='Layer numbers.')
        parser.add_argument('--n_intents', type=int, default=128, help='Number of latent intents')
        parser.add_argument('--temp', type=float, default=0.2, help='temperature in ssl loss')
        parser.add_argument('--show_step', type=int, default=10, help='Test every show_step epochs.')
        parser.add_argument('--Ks', nargs='?', default='[20, 40]', help='Metrics scale')
        parser.add_argument('--dropout', type=float, default=0.1, help='Regularizations.')

    # Test-Recall=[0.1948, 0.2753], Test-NDCG =[0.1163, 0.1373]
    # drop 0.1 Test-Recall=[0.1990, 0.2787], Test-NDCG =[0.1189, 0.1397] 丢一个1994
    # 加特征掩码 Test-Recall=[0.2005, 0.2757], Test-NDCG =[0.1220, 0.1417]
    #不加1除2 Test-Recall=[0.2035, 0.2772], Test-NDCG =[0.1232, 0.1425]
    # 全局掩码Test-Recall=[0.2058, 0.2860], Test-NDCG =[0.1229, 0.1440]
    # +去噪图 Test-Recall=[0.2109, 0.2919], Test-NDCG =[0.1259, 0.1472]
    # -两个对比学习Test-Recall=[0.1515, 0.2199], Test-NDCG =[0.0901, 0.1079]
    # +嵌入对比0.1Test-Recall=[0.2095, 0.2887], Test-NDCG =[0.1259, 0.1467]
    elif dataset=='gowalla':
        parser = argparse.ArgumentParser(description="Run MDGCL.")
        parser.add_argument('--data_path', nargs='?', default='data/', help='Input data path.')
        parser.add_argument('--seed', type=int, default=1024, help='random seed')
        parser.add_argument('--dataset', nargs='?', default=dataset,help='Choose a dataset from {gowalla, amazon, tmall}')
        parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
        parser.add_argument('--save_model', type=bool, default=False, help='Whether to save')
        parser.add_argument('--epoch', type=int, default=200, help='Number of epochs')
        parser.add_argument('--embed_size', type=int, default=32, help='Embedding size.')
        parser.add_argument('--n_batch', type=int, default=40, help='Number of mini-batches')
        parser.add_argument('--batch_size', type=int, default=10240, help='batch size')
        parser.add_argument('--train_num', type=int, default=10000, help='Number of training instances per epoch')
        parser.add_argument('--sample_num', type=int, default=40, help='Number of pos/neg samples for each instance')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
        parser.add_argument('--emb_reg', type=float, default=1e-4, help='Regularizations.')
        parser.add_argument('--cen_reg', type=float, default=1e-4, help='Regularizations.')
        parser.add_argument('--ssl_reg', type=float, default=1e-0, help='Reg weight for ssl loss')
        parser.add_argument('--n_layers', type=int, default=2, help='Layer numbers.')
        parser.add_argument('--n_intents', type=int, default=128, help='Number of latent intents')
        parser.add_argument('--temp', type=float, default=0.2, help='temperature in ssl loss')
        parser.add_argument('--show_step', type=int, default=10, help='Test every show_step epochs.')
        parser.add_argument('--Ks', nargs='?', default='[20, 40]', help='Metrics scale')
        parser.add_argument('--dropout', type=float, default=0.1, help='Regularizations.')

    # t0.2 [0.0873, 0.1336], Test-NDCG =[0.0663, 0.0815]
    #drop 0.1 Test - Recall = [0.0891, 0.1347], Test - NDCG = [0.0678, 0.0828]
    # Test-Recall=[0.0904, 0.1353], Test-NDCG =[0.0698, 0.0845]
    # Test-Recall=[0.0903, 0.1353], Test-NDCG =[0.0695, 0.0843]
    # 全局掩码 Test-Recall=[0.0945, 0.1418], Test-NDCG =[0.0723, 0.0879]
    # +去噪图 Test-Recall=[0.0959, 0.1445], Test-NDCG =[0.0727, 0.0887]
    elif dataset=='amazon':
        parser = argparse.ArgumentParser(description="Run MDGCL.")
        parser.add_argument('--data_path', nargs='?', default='data/', help='Input data path.')
        parser.add_argument('--seed', type=int, default=1024, help='random seed')
        parser.add_argument('--dataset', nargs='?', default=dataset,help='Choose a dataset from {gowalla, amazon, tmall}')
        parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
        parser.add_argument('--save_model', type=bool, default=False, help='Whether to save')
        parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
        parser.add_argument('--embed_size', type=int, default=32, help='Embedding size.')
        parser.add_argument('--n_batch', type=int, default=40, help='Number of mini-batches')
        parser.add_argument('--batch_size', type=int, default=10240, help='batch size')
        parser.add_argument('--train_num', type=int, default=10000, help='Number of training instances per epoch')
        parser.add_argument('--sample_num', type=int, default=40, help='Number of pos/neg samples for each instance')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
        parser.add_argument('--emb_reg', type=float, default=1e-4, help='Regularizations.')
        parser.add_argument('--cen_reg', type=float, default=1e-4, help='Regularizations.')
        parser.add_argument('--ssl_reg', type=float, default=1e-0, help='Reg weight for ssl loss')
        parser.add_argument('--n_layers', type=int, default=2, help='Layer numbers.')
        parser.add_argument('--n_intents', type=int, default=128, help='Number of latent intents')
        parser.add_argument('--temp', type=float, default=0.2, help='temperature in ssl loss')
        parser.add_argument('--show_step', type=int, default=30, help='Test every show_step epochs.')
        parser.add_argument('--Ks', nargs='?', default='[20, 40]', help='Metrics scale')
        parser.add_argument('--dropout', type=float, default=0.1, help='Regularizations.')

    elif dataset=='mooc':
        parser = argparse.ArgumentParser(description="Run MDGCL.")
        parser.add_argument('--data_path', nargs='?', default='data/', help='Input data path.')
        parser.add_argument('--seed', type=int, default=1024, help='random seed')
        parser.add_argument('--dataset', nargs='?', default=dataset,help='Choose a dataset from {gowalla, amazon, tmall}')
        parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
        parser.add_argument('--save_model', type=bool, default=False, help='Whether to save')
        parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
        parser.add_argument('--embed_size', type=int, default=32, help='Embedding size.')
        parser.add_argument('--n_batch', type=int, default=40, help='Number of mini-batches')
        parser.add_argument('--batch_size', type=int, default=10240, help='batch size')
        parser.add_argument('--train_num', type=int, default=10000, help='Number of training instances per epoch')
        parser.add_argument('--sample_num', type=int, default=40, help='Number of pos/neg samples for each instance')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
        parser.add_argument('--emb_reg', type=float, default=1e-4, help='Regularizations.')
        parser.add_argument('--cen_reg', type=float, default=1e-4, help='Regularizations.')
        parser.add_argument('--ssl_reg', type=float, default=1e-1, help='Reg weight for ssl loss')
        parser.add_argument('--n_layers', type=int, default=2, help='Layer numbers.')
        parser.add_argument('--n_intents', type=int, default=128, help='Number of latent intents')
        parser.add_argument('--temp', type=float, default=0.2, help='temperature in ssl loss')
        parser.add_argument('--show_step', type=int, default=30, help='Test every show_step epochs.')
        parser.add_argument('--Ks', nargs='?', default='[5,10]', help='Metrics scale')
        parser.add_argument('--dropout', type=float, default=0.1, help='Regularizations.')

    #Test-Recall=[0.0730, 0.1133], Test-NDCG =[0.0514, 0.0655]
    #Test - Recall = [0.0723, 0.1116], Test - NDCG = [0.0514, 0.0649]都一样
    # Test-Recall=[0.0760, 0.1156], Test-NDCG =[0.0538, 0.0676] Emask
    # Test-Recall=[0.0752, 0.1164], Test-NDCG =[0.0532, 0.0675] Sigmoid
    # Test - Recall = [0.0761, 0.1185], Test - NDCG = [0.0534, 0.0681]
    elif dataset=='tmall':
        parser = argparse.ArgumentParser(description="Run MDGCL.")
        parser.add_argument('--data_path', nargs='?', default='data/', help='Input data path.')
        parser.add_argument('--seed', type=int, default=1024, help='random seed')
        parser.add_argument('--dataset', nargs='?', default=dataset,help='Choose a dataset from {gowalla, amazon, tmall}')
        parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
        parser.add_argument('--save_model', type=bool, default=False, help='Whether to save')
        parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
        parser.add_argument('--embed_size', type=int, default=32, help='Embedding size.')
        parser.add_argument('--n_batch', type=int, default=40, help='Number of mini-batches')
        parser.add_argument('--batch_size', type=int, default=10240, help='batch size')
        parser.add_argument('--train_num', type=int, default=10000, help='Number of training instances per epoch')
        parser.add_argument('--sample_num', type=int, default=40, help='Number of pos/neg samples for each instance')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
        parser.add_argument('--emb_reg', type=float, default=1e-4, help='Regularizations.')
        parser.add_argument('--cen_reg', type=float, default=1e-4, help='Regularizations.')
        parser.add_argument('--ssl_reg', type=float, default=1e-0, help='Reg weight for ssl loss')
        parser.add_argument('--n_layers', type=int, default=2, help='Layer numbers.')
        parser.add_argument('--n_intents', type=int, default=128, help='Number of latent intents')
        parser.add_argument('--temp', type=float, default=0.2, help='temperature in ssl loss')
        parser.add_argument('--show_step', type=int, default=10, help='Test every show_step epochs.')
        parser.add_argument('--Ks', nargs='?', default='[20, 40]', help='Metrics scale')
        parser.add_argument('--dropout', type=float, default=0.1, help='Regularizations.')
    else:
        parser = argparse.ArgumentParser(description="Run MDGCL.")
        parser.add_argument('--data_path', nargs='?', default='data/', help='Input data path.')
        parser.add_argument('--seed', type=int, default=1024, help='random seed')
        parser.add_argument('--dataset', nargs='?', default=dataset,help='Choose a dataset from {gowalla, amazon, tmall}')
        parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
        parser.add_argument('--save_model', type=bool, default=False, help='Whether to save')
        parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
        parser.add_argument('--embed_size', type=int, default=32, help='Embedding size.')
        parser.add_argument('--n_batch', type=int, default=40, help='Number of mini-batches')
        parser.add_argument('--batch_size', type=int, default=10240, help='batch size')
        parser.add_argument('--train_num', type=int, default=10000, help='Number of training instances per epoch')
        parser.add_argument('--sample_num', type=int, default=40, help='Number of pos/neg samples for each instance')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
        parser.add_argument('--emb_reg', type=float, default=1e-4, help='Regularizations.')
        parser.add_argument('--cen_reg', type=float, default=1e-4, help='Regularizations.')
        parser.add_argument('--ssl_reg', type=float, default=1e-1, help='Reg weight for ssl loss')
        parser.add_argument('--n_layers', type=int, default=2, help='Layer numbers.')
        parser.add_argument('--n_intents', type=int, default=128, help='Number of latent intents')
        parser.add_argument('--temp', type=float, default=0.2, help='temperature in ssl loss')
        parser.add_argument('--show_step', type=int, default=30, help='Test every show_step epochs.')
        parser.add_argument('--Ks', nargs='?', default='[20, 40]', help='Metrics scale')
        parser.add_argument('--dropout', type=float, default=0.1, help='Regularizations.')

    return parser.parse_args()
