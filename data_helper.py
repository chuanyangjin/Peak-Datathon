import json
import random
from tkinter import FALSE, N
import zipfile
from io import BytesIO
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
)
# from transformers import BertTokenizer, AutoTokenizer, AutoConfig
# from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from PIL import Image
# from category_id_map import category_id_to_lv2id, category_id_to_lv1id
from tqdm import tqdm
import re, math
from torch.utils.data.distributed import DistributedSampler as DS

# from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import scipy
import time, os
import torch.distributed as dist
import torch.utils.data.distributed
from tqdm import tqdm

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = -1

import pandas as pd
import random
from joblib import Parallel, delayed
import logging
import pickle

# from util import cache_wrapper
# from util_func.hdf5_getters import open_h5_file_read, get_segments_timbre
import sqlite3
import json
import random


def create_dataloaders(train_data_ratio, batch_size):

    # if 'TPS' not in args.config['train_csv']:
    # 
    # else:
    if os.path.exists("cache/dataset_neg3.pkl"):
        logging.info("Loaded cached dataset")
        with open("cache/dataset_neg3.pkl", 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = CustomDataset()
        with open("cache/dataset_neg3.pkl", 'wb') as f:
            pickle.dump(dataset, f)
        logging.info("Cache dataset")

    all_ids = np.arange(len(dataset))
    np.random.shuffle(all_ids)
    val_index = int((1-train_data_ratio) * len(dataset))
    val_ids = all_ids[:val_index]

    # val_ids = np.random.choice(len(dataset), int(0.2 * len(dataset)))
    with open("val_ids.json", 'w') as f:
        f.write(json.dumps(val_ids.tolist()))
    train_ids = list(set(np.arange(len(dataset))) - set(val_ids))
    logging.info("Number of train set %d, number of valid set %d" %(len(train_ids), len(val_ids)))

    train_dataset = Subset(dataset, train_ids)
    val_dataset = Subset(dataset, val_ids)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size * 4,
        shuffle=False,
        pin_memory=False
    )


    return train_dataloader, val_dataloader, train_dataset, val_dataset

def user_idx_generator(n_users, batch_users):
    ''' helper function to generate the user index to loop through the dataset
    '''
    for start in range(0, n_users, batch_users):
        end = min(n_users, start + batch_users)
        yield slice(start, end)
def get_n_neg_example(
        neg_candid_map,
        item_ids, 
        uid, neg_num, times_instance_num):

    pos_song_ids = neg_candid_map[uid]
    neg_candidates = item_ids - pos_song_ids
    N = len(pos_song_ids)

    if times_instance_num:
        neg_examples = random.sample(neg_candidates, neg_num * N)
    else:
        neg_examples = random.sample(neg_candidates, neg_num)

    return [uid] * len(neg_examples), neg_examples
def get_n_neg_example_helper(
        neg_candid_map,
        item_ids, 
        user_idx, 
        users_ids, neg_num, times_instance_num):
    uids, negs = [], []
    for uid in users_ids[user_idx]:
        uid, neg = get_n_neg_example(neg_candid_map, item_ids, uid, neg_num, times_instance_num)
        uids += uid
        if not times_instance_num:
            negs.append(neg)
        else:
            negs += neg
    return uids,negs
    # return [uid] * len(neg_examples), neg_candidates

class CustomDataset(Dataset):
    def __init__(self, neg_sample = 3):
        super().__init__()

        order_df = pd.read_csv('data/olist_orders_dataset.csv')
        order_item_df = pd.read_csv('data/olist_order_items_dataset.csv')
        # prod2order = pd.Series(order_item_df.product_id.values,index=order_item_df.order_id).to_dict()

        df = pd.merge(order_df, order_item_df, how="inner", on='order_id')
        # order2cust = pd.Series(order_df.order_id.values,index=order_df.customer_id).to_dict()
        all_customers = order_df.customer_id
        cust2id = {v:i for i,v in enumerate(all_customers.unique())}
        all_products = order_item_df.product_id
        prod2id = {v:i for i,v in enumerate(all_products.unique())}

        self.uniques = [len(cust2id), len(prod2id)]
        print(f"There are {self.uniques[0]} customers and {self.uniques[1]} products")

        df.customer_id = df.customer_id.map(cust2id)
        df.product_id = df.product_id.map(prod2id)

        # print(len(df), len(order_item_df), len(order_df))
        # print(df[['customer_id', 'product_id']].head())

        self.custs = df['customer_id'].to_list()
        self.prods = df['product_id'].to_list()
        self.labels = [1] * len(self.prods)

        # Negative sampling
        for cust in tqdm(all_customers):
            bought = df[df.customer_id == cust]['product_id']
            no_purchase = set(all_products) - set(bought)
            # print(cust2id[cust])
            self.custs += [cust2id[cust]] * neg_sample
            # print(cust2id[cust])
            self.prods += list(map(lambda x:prod2id[x], random.sample(no_purchase, neg_sample)))
            self.labels += [0] * neg_sample

    
    def __len__(self):
        return len(self.custs)
    
    def __getitem__(self, index):
        return [self.custs[index], self.prods[index], self.labels[index]]

    def get_num_of_unique(self):
        return self.uniques

if __name__ == "__main__":
    dataset = CustomDataset(3)
    with open("cache/dataset_neg3.pkl", 'wb') as f:
        pickle.dump(dataset, f)