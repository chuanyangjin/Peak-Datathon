import logging
import random

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn import metrics

import torch
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import os
import pickle
import logging

import logging
import os
import time
from pandas import concat
import torch

# from model import MultiModal
# from models import *

# from new_model import MultiModal
# from config import parse_args
from data_helper import create_dataloaders
# from apex import amp
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch.optim as optim

import numpy as np
from tqdm import tqdm

     
def build_optimizer(lr, max_epochs, model, batch_size, train_dataset):
    optimizer = torch.optim.SparseAdam(
        model.parameters(), lr=lr
    )

    # Calculate scheduler number
    data_size = len(train_dataset)

    max_steps = data_size * max_epochs // batch_size
    warmup_steps = max_steps // 10
    print("max steps: " + str(max_steps))

    print_steps = max_steps / max_epochs // 10

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
    )
    return optimizer, scheduler


def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    
def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

def evaluate(predictions, labels, metrics):
    eval_resuts = dict()

    if 'auc' in metrics:
        # prediction and labels are all level-2 class ids
        fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        eval_resuts['auc'] = auc
    return eval_resuts


def cal_hit(gt_index, pred_indices):
    h = (pred_indices == gt_index).sum(dim=0).float().mean()
    assert h <= 1
    return h


def cal_ndcg(gt_index, pred_items):
    # 
    index = (pred_items==gt_index).nonzero(as_tuple=True)[1]
    
    ndcg = np.reciprocal(np.log2(index+2))

    return ndcg.sum()/pred_items.size()[1]



def validate(model, val_dataloader, device,):
    model.eval()
    predictions = []
    labels = []
    losses = []
    uids = []
    iids = []

    hr, ncdg = [], []
    loss_fnt = F.binary_cross_entropy_with_logits

    with torch.no_grad():
        for batch in (val_dataloader):

            uid, iid, label = [i.to(device) for i in batch]

            loss = model(uid, iid, label)
            # label = label.float()
            # print(preds)
            # loss = loss_fnt(preds, label) 
            losses.append(loss.cpu().numpy())

            uids.extend(uid)
            iids.extend(iid)
            # predictions.extend(preds.cpu().numpy())
            labels.extend(label.cpu().numpy())


    metrics = []
    # results = evaluate(predictions, labels, metrics)
    results = {}

    model.train()
    return np.mean(losses), results




def train_and_validate(args):
    # 1. load data
    #
    #   train_dataset[0] = [uid, sid, ratings, features]
    #
    train_dataloader, val_dataloader, train_dataset, val_dataset = create_dataloaders(args)

    # 2. build model and optimizers
    user_num, item_num = train_dataset.dataset.get_num_of_unique()
    args.config['model_config'].update( {'user_num': user_num, 'item_num': item_num} )

    print("Num users: %d, num items %d" % (user_num, item_num))
    model_class = eval(args.config['model_type'])
    model = model_class(**args.config['model_config'])



    if False and os.path.exists(args.ckpt_file):
        pass

    else:
        print("No ckpt file loaded")

    optimizer, scheduler = build_optimizer(args, model, train_dataset)
    if args.device == 'cuda':

        # FP16
        # model, optimizer = amp.initialize(
        #     model.to(args.device), optimizer, enabled=args.fp16, opt_level='O1',
        #     # keep_batchnorm_fp32=True
        # )
        # model = torch.nn.parallel.DataParallel(model)
        model.to(args.device)


    

def train(model, optimizer, scheduler, train_dataloader, val_dataloader, max_epochs, device):
    # 3. training
    step = 0
    # best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * max_epochs
    print_steps = num_total_steps // 10 // max_epochs
    # print_steps = 10

    loss, results = validate(model, val_dataloader, device)
    results = {k: round(v, 4) for k, v in results.items()}
    res_str = " ".join([f"{k}: {v}" for k,v in results.items()])
    print(f"Validation Set - step {step}: loss {loss:.3f} " + res_str)

    loss_fnt = F.binary_cross_entropy_with_logits
    for epoch in range(max_epochs):
        for batch in train_dataloader:
            model.train()
            # print(batch)
            uid, iid, neg = [i.to(device) for i in batch]

            loss = model(uid, iid, neg)
            # print(preds)
            # exit(0)
            # labels = label.float()
            # loss = loss_fnt(preds, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                print(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, \
                             learning_rate {scheduler.get_last_lr()[0]}")

        # 4. validation
        loss, results = validate(model, val_dataloader, device)
        results = {k: round(v, 4) for k, v in results.items()}
        res_str = " ".join([f"{k}: {v}" for k,v in results.items()])
        print("="*5 + "Epoch "+str(epoch) + "="*5)
        print(f"Validation Set - step {step}: loss {loss:.3f} " + res_str)

        # 5. save checkpoint
        # mean_f1 = results['mean_f1']
        if False and epoch >= args.config['train_config']['max_epochs'] - 1: # ncdg > best_score:
            best_score = loss
            state_dict = model.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'loss': loss},
                       f'{args.savedmodel_path}/model_{args.config["experiment_name"]}_{args.config["model_type"]}_trainRatio_{args.train_data_ratio}_epoch_{epoch}_loss_{loss}.bin')

