
import argparse
import os
import time
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
def get_file_info(directory):
    file_info_list = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            parent_dir = os.path.basename(os.path.dirname(file_path))
            grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            file_info_list.append((grandparent_dir, parent_dir, filename))
    return file_info_list
def main(m_,seq_len_,tag,pred_l,data_type,process_mode):
    fix_seed = 2023
    torch.manual_seed(fix_seed)
    random.seed(fix_seed)

    np.random.seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--prob_forecasting', action='store_true', help='using probabilistic forecasting')
    parser.add_argument('--scales', default=[16, 8, 4, 2, 1], help='scales in mult-scale')
    parser.add_argument('--scale_factor', type=int, default=2, help='scale factor for upsample')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    # Common Model parameters.
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=128)
    parser.add_argument('-d_v', type=int, default=128)
    parser.add_argument('-d_bottleneck', type=int, default=128)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layer', type=int, default=4)

    args = parser.parse_args()

    args.loss = 'mse'

    args.use_multi_gpu = False
    args.train_epochs=30
    args.itr=1
    args.gpu = 0
    args.devices='1'

    args.script_id = ''
    args.train_only = False
    args.d_model=512

    args.dived=True

    args.loss = 'mse'

    args.data_type=data_type
    args.dir_id='_0'
    args.data_path_list = []


    args.data = 'Fund'
    if args.data_type=='Fund_all':
        data_path='./dataset/'+'Fund_all'
        args.root_path=data_path
    elif args.data_type=='Fund_all_2':
        data_path='./dataset/'+'Fund_all_2'
        args.root_path=data_path

    args.data_path_list=os.listdir(data_path)

    args.model = m_


    args.checkpoints = './checkpoints_' + args.data_type + '_' + str(pred_l) + '/' + args.model + '/'
    seq_len=seq_len_
    args.test_point_num = 67

    args.preprocess_data=True

    args.seq_len = seq_len

    args.embed_id_size = 64
    args.learning_rate = 1e-4
    args.cal_scaler_global = False

    args.p_hidden_dims=[128, 128,128,128]
    args.p_hidden_layers=4
    args.enc_in=args.dec_in=2
    args.c_out=2


    args.train_epochs = 30
    args.wmape=True

    if args.wmape:
        args.loss_real = 'wmape'
    else:
        args.loss_real = 'mse'
    c = 2
    args.batch_size = 128

    args.e_layers = 4
    args.gpu=5
    args.device='cuda:'+str(args.gpu)
    args.enc_in = c
    args.dec_in = c
    args.c_out = c
    args.seq_len=seq_len_
    args.label_len=10
    args.D_norm=True
    args.sel_scale=process_mode[-1]
    args.scale_or=process_mode[0]
    args.scale_adaptive=process_mode[1]
    args.normalization = process_mode[2]

    args.scale_adaptive_pow=False
    if args.scale_or or args.scale_adaptive:
        args.no_scale=False
    else:
        args.no_scale = True
    args.back=args.seq_len
    args.pred_len = pred_l
    if args.no_scale and args.normalization:
        tag = tag + 'stand_noscale'
    elif args.no_scale:
        tag = tag + 'noscale_nonorm_real'

    elif args.scale_or:
        tag = tag + 'scale'

    elif args.scale_adaptive:
        tag = tag + 'scale_adaptive'
        if args.sel_scale:

            tag = tag + '_sel_scale_average'
    args.itr=1
    args.is_training=True
    args.record=True
    if args.model == 'Pyraformer':
        args.input_size = args.seq_len
        args.predict_step = args.pred_len

    if args.label_len>args.seq_len:
        args.label_len=args.seq_len
    if args.model == 'DeepAr':
        args.label_len = 0
    print('Args in experiment:')
    print(args)

    if args.prob_forecasting:
        assert args.loss == 'mse'

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            if tag != '':
                setting = f'{args.data_type}_{args.model}_{args.seq_len}_{args.pred_len}_{args.loss_real}_{tag}'
            else:
                setting = f'{args.data_type}_{args.model}_{args.seq_len}_{args.pred_len}_{args.loss_real}'
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0

        if tag != '':
            setting = f'{args.data_type}_{args.model}_{args.seq_len}_{args.pred_len}_{args.loss_real}_{tag}'
        else:
            setting = f'{args.data_type}_{args.model}_{args.seq_len}_{args.pred_len}_{args.loss_real}'
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
if __name__ == "__main__":
    tag=['']
    data_type_all = ['Fund_all','Fund_all_2']
    pred_len = [5,10]
    seq_len_all = [30]
    model_all_F = ['Transformer','Autoformer','Informer','Performer',]
    #VS: Vanilla scale scaling
    #SC: only using our scale calibrating sub-module
    #SS+SC: Equivalent to out AS module, using both our scale calibrating and scaling selection sub-modules.
    #         no_scale_no_norm              normalization                VS                      SC                     SS+SC
    mode=[[False,False,False,False],[False,False,True,False],[True,False,False,False],[False,True,False,False],[False,True,False,True]]
    # mode = [[False, False, False,False]]
    # mode = [[False,False,True,False]]
    # mode = [[True,False,False,False]]
    # mode=[[False,True,False,False]]
    # mode = [[False,True,False,True]]

    for ta_g in tag:
        for data_type in data_type_all:
            for pred_l in pred_len:
                for m_ in model_all_F:
                    for seq_len in seq_len_all:
                        for process_mode in mode:
                            main(m_, seq_len, ta_g, pred_l, data_type, process_mode)
                            torch.cuda.empty_cache()

