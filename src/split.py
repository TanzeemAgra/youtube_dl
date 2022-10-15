import os
import shutil
import random
import yaml
import argparse
import numpy as np
import pandas as pd
from get_data import get_data

def train_and_test(config):
    config=get_data(config)
    root_dir=config['data_source']['data_src']
    dest=config['load_data']['preprocessed_data']
    p=config['load_data']['full_Path']
    cla=config['data_source']['data_src']
    cla=os.listdir(cla)
    print(cla)
    splitr=config['train_split']['split_ratio']
    print(splitr)
    for k in range(len(cla)):
        print(cla[k])
        per=len(os.listdir((os.path.join(root_dir, cla[k]))))
        print(per)
        cnt = 0
        for j in os.listdir(os.path.join(root_dir, cla[k])):
            pat=os.path.join(p + '/' +cla[k],j )
            #print(pat)
            split_ratio=round((splitr/100)*per)
            #print(split_ratio)
            if cnt != split_ratio:
                #print(cnt)
                shutil.copy(pat, dest+'/'+'train/class_'+str(k))
                cnt = cnt + 1
            else:
                shutil.copy(pat, dest+'/'+'test/class_'+str(k))

    print('done')
    


if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--config", default='params.yaml')
    passed_args=args.parse_args()
    train_and_test(config=passed_args.config)