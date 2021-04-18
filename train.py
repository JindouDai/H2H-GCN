from __future__ import division
from __future__ import print_function
import datetime
import json
import logging
import os
import pickle
import time
import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics
from manifolds.StiefelManifold import StiefelManifold
from utils.pre_utils import *
import warnings
warnings.filterwarnings('ignore')

def train(args):
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    data = load_data(args, os.path.join('./data', args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        else:
            Model = RECModel
            args.eval_freq = args.epochs + 1

    # Model and optimizer
    model = Model(args)
    
    optimizer, lr_scheduler, stiefel_optimizer, stiefel_lr_scheduler = \
                        set_up_optimizer_scheduler(False, args, model, args.lr, args.lr_stie)
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])

    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)

    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    for epoch in range(args.epochs):       
        t = time.time()
        model.train()
        optimizer.zero_grad()
        stiefel_optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['hgnn_adj'], data['hgnn_weight']) 
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        stiefel_optimizer.step()
        lr_scheduler.step()
        stiefel_lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {:04f}, stie_lr: {:04f}'.format(lr_scheduler.get_lr()[0], stiefel_lr_scheduler.get_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['hgnn_adj'], data['hgnn_weight'])
            for i in range(embeddings.size(0)):
                if (embeddings[i] != embeddings[i]).sum()>1:
                    print('PART train  i', i, 'embeddings[i]', embeddings[i])
            val_metrics = model.compute_metrics(embeddings, data, 'val') 
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics): 
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")
                    break

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    if args.save:
        np.save(os.path.join(save_dir, str(args.dataset)+'_embeddings.npy'), best_emb.cpu().detach().numpy())
        torch.save(model.state_dict(), os.path.join(save_dir, str(args.dataset)+'model_auc' + str(best_rocauc) + '.pth'))
        logging.info(f"Saved model in {save_dir}")
    if args.task == 'lp':
        return best_test_metrics['roc']
    if args.task == 'nc':
        return best_test_metrics['f1']

def cal_std(acc):
    if acc[0] < 1:
        for i in range(len(acc)):
            acc[i] = acc[i] * 100
    mean = np.mean(acc)
    var = np.var(acc)
    std = np.std(acc)
    return mean, std

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if self.model == 'H2HGCN':
        args.dim = args.dim + 1
    result_list = []
    for i in range(10):
        args = parser.parse_args()
        args.weight_manifold = StiefelManifold(args, 1)
        args.stie_vars = []
        args.eucl_vars = []
        # set_seed(int(time.time()))
        set_seed(args.seed)
        result = train(args)
        result_list.append(result)
        print(result_list)
    mean, std = cal_std(result_list)
    print('mean:', mean, 'std:', std)
