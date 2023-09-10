import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from data_provider.data_loader import Dataset_Custom,Dataset_Fund
from torch.utils.data import DataLoader
import pandas as pd
import os
import numpy as np
data_dict = {
    'custom': Dataset_Custom,
    'Fund':Dataset_Fund,
}

def obtain_max_scaler(args):

    mean_all={}
    for data_path in args.data_path_list:
        df_raw = pd.read_csv(os.path.join(args.root_path,
                                          data_path))
        if 'Fund' in args.data_type:
            df_raw_x = df_raw[['apply_amt', 'redeem_amt']]
            mean=np.mean(np.abs(np.mean(np.abs(df_raw_x.to_numpy()),axis=0)))
            mean_all[data_path]=mean
        elif 'elect' in args.data_type:
            df_raw_x = df_raw['consumption']

            mean=np.mean(np.abs(np.mean(np.abs(df_raw_x.to_numpy()),axis=0)))
            # print(mean)
            # time.sleep(500)
            mean_all[data_path]=mean

    sorted_mean_all = dict(sorted(mean_all.items(), key=lambda item: item[1], reverse=True))
    args.sorted_mean_all=sorted_mean_all

    df_raw = pd.read_csv(os.path.join(args.root_path,
                                      list(sorted_mean_all.keys())[0]))

    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test

    border1s = [0, num_train - args.seq_len, len(df_raw) - num_test - args.seq_len]
    border2s = [num_train, num_train + num_vali, len(df_raw)]
    type_map = {'train': 0, 'val': 1, 'test': 2}
    set_type = type_map['train']

    border1 = border1s[set_type]
    border2 = border2s[set_type]
    if 'Fund' in args.data_type:
        df_raw_x = df_raw[['apply_amt', 'redeem_amt']]
    else:
        df_raw_x = df_raw[['consumption']]
    train_data = df_raw_x[border1:border2]
    scaler=MinMaxScaler()
    scaler.fit(train_data.values)
    return scaler

def data_provider(args, flag):

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    if args.state=='data_process':
        shuffle_flag = False
        drop_last=False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    print(flag,shuffle_flag,drop_last)
    # time.sleep(500)
    cal_scaler=args.cal_scaler_global
    args.flag=flag
    args.scaler_custom=None
    print(args.state)
    if args.state == 'data_process':
        if cal_scaler:
            args.scaler_custom=obtain_max_scaler(args)
        data_set_all=[]
        data_loader_all=[]

        all_efective_dataset=[]
        for data_path in args.data_path_list:

            all_efective_dataset.append(data_path)
            data_set = Data(
                root_path=args.root_path,
                data_path=data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq, args=args
            )

            drop_last = False

            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=0,  # args.num_workers
                drop_last=drop_last)
            data_set_all.append(data_set)
            data_loader_all.append(data_loader)

        return data_set_all,data_loader_all
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,args=args
        )
        drop_last=False
        # print(flag, len(data_set),batch_size)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=0, #args.num_workers
            drop_last=drop_last)
        return data_set, data_loader
