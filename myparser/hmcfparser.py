import argparse

def parse_args(dataset):
    print('HMCF')
    if dataset=='amazon':
        parser = argparse.ArgumentParser(description="Run New MDGCL.")
        parser.add_argument('--data_path', nargs='?', default='data/', help='Input data path.')
        parser.add_argument('--seed', type=int, default=2023, help='random seed')
        parser.add_argument('--dataset', nargs='?', default=dataset, help='{gowalla, amazon, tmall}')
        parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
        parser.add_argument('--save_model', type=bool, default=False, help='Whether to save')
        parser.add_argument('--epoch', type=int, default=300, help='Number of epochs')
        parser.add_argument('--embed_size', type=int, default=32, help='Embedding size.')
        parser.add_argument('--n_batch', type=int, default=40, help='Number of mini-batches')
        parser.add_argument('--batch_size', type=int, default=10240, help='batch size')
        parser.add_argument('--train_num', type=int, default=10000, help='Number of training instances per epoch')
        parser.add_argument('--sample_num', type=int, default=40, help='Number of pos/neg samples for each instance')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. tune{1e-3,1e-4}')
        parser.add_argument('--emb_reg', type=float, default=2.5e-5, help='Regularizations. tune{1e-4,1e-5,1e-6}')
        parser.add_argument('--n_layers', type=int, default=2, help='Layer numbers.')
        parser.add_argument('--ssl_reg', type=float, default=1e-1, help='Reg weight for ssl loss tune{1e-1,1e-2}')
        parser.add_argument('--lhyper', type=float, default=1e-1, help='lambda2.')
        parser.add_argument('--temp', type=float, default=0.2, help='temperature in ssl loss tune{0.1,0.2,0.4}')
        parser.add_argument('--dropout', type=float, default=0.5, help='rou')
        parser.add_argument('--show_step', type=int, default=10, help='Test every show_step epochs.')
        parser.add_argument('--Ks', nargs='?', default='[20, 40]', help='Metrics scale')

    elif dataset == 'tmall':
        parser = argparse.ArgumentParser(description="Run New MDGCL.")
        parser.add_argument('--data_path', nargs='?', default='data/', help='Input data path.')
        parser.add_argument('--seed', type=int, default=2023, help='random seed')
        parser.add_argument('--dataset', nargs='?', default=dataset, help='{gowalla, amazon, tmall}')
        parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
        parser.add_argument('--save_model', type=bool, default=False, help='Whether to save')
        parser.add_argument('--epoch', type=int, default=60, help='Number of epochs')
        parser.add_argument('--embed_size', type=int, default=32, help='Embedding size.')
        parser.add_argument('--n_batch', type=int, default=40, help='Number of mini-batches')
        parser.add_argument('--batch_size', type=int, default=10240, help='batch size')
        parser.add_argument('--train_num', type=int, default=10000, help='Number of training instances per epoch')
        parser.add_argument('--sample_num', type=int, default=40, help='Number of pos/neg samples for each instance')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. tune{1e-3,1e-4}')
        parser.add_argument('--emb_reg', type=float, default=2.5e-5, help='Regularizations. tune{1e-4,1e-5,1e-6}')
        parser.add_argument('--n_layers', type=int, default=2, help='Layer numbers.')
        parser.add_argument('--ssl_reg', type=float, default=1e-1, help='Reg weight for ssl loss tune{1e-1,1e-2}')
        parser.add_argument('--lhyper', type=float, default=1e-1, help='lambda2.')
        parser.add_argument('--temp', type=float, default=0.2, help='temperature in ssl loss tune{0.1,0.2,0.4}')
        parser.add_argument('--dropout', type=float, default=0.5, help='rou')
        parser.add_argument('--show_step', type=int, default=10, help='Test every show_step epochs.')
        parser.add_argument('--Ks', nargs='?', default='[20, 40]', help='Metrics scale')



    else:
        parser = argparse.ArgumentParser(description="Run New MDGCL.")
        parser.add_argument('--data_path', nargs='?', default='data/', help='Input data path.')
        parser.add_argument('--seed', type=int, default=2023, help='random seed')
        parser.add_argument('--dataset', nargs='?', default=dataset, help='{gowalla, amazon, tmall}')
        parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
        parser.add_argument('--save_model', type=bool, default=False, help='Whether to save')
        parser.add_argument('--epoch', type=int, default=300, help='Number of epochs')
        parser.add_argument('--embed_size', type=int, default=32, help='Embedding size.')
        parser.add_argument('--n_batch', type=int, default=40, help='Number of mini-batches')
        parser.add_argument('--batch_size', type=int, default=10240, help='batch size')
        parser.add_argument('--train_num', type=int, default=10000, help='Number of training instances per epoch')
        parser.add_argument('--sample_num', type=int, default=40, help='Number of pos/neg samples for each instance')
       parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. tune{1e-3,1e-4}')
        parser.add_argument('--emb_reg', type=float, default=2.5e-5, help='Regularizations. tune{1e-4,1e-5,1e-6}')
        parser.add_argument('--n_layers', type=int, default=2, help='Layer numbers.')
        parser.add_argument('--ssl_reg', type=float, default=1e-1, help='Reg weight for ssl loss tune{1e-1,1e-2}')
        parser.add_argument('--lhyper', type=float, default=1e-1, help='lambda2.')
        parser.add_argument('--temp', type=float, default=0.2, help='temperature in ssl loss tune{0.1,0.2,0.4}')
        parser.add_argument('--dropout', type=float, default=0.5, help='rou')
        parser.add_argument('--show_step', type=int, default=10, help='Test every show_step epochs.')
        parser.add_argument('--Ks', nargs='?', default='[20, 40]', help='Metrics scale')

    return parser.parse_args()
