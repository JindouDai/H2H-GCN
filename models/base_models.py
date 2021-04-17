import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layers import FermiDiracDecoder
import manifolds
import models.encoders as encoders
from models.encoders import H2HGCN
from models.decoders import model2decoder
from utils.eval_utils import acc_f1
from manifolds import LorentzManifold

 
class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.c = torch.Tensor([1.]).cuda().to(args.device)
        args.manifold = self.manifold = LorentzManifold(args)
        args.feat_dim = args.feat_dim + 1
        # add 1 for Lorentz as the degree of freedom is d - 1 with d dimensions
        args.dim = args.dim + 1
        self.nnodes = args.n_nodes
        self.encoder = H2HGCN(args, 1)

    def encode(self, x, hgnn_adj, hgnn_weight):
        h = self.encoder.encode(x, hgnn_adj, hgnn_weight)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)


    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}'] 
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges

    def decode(self, h, idx, split):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c) 
        probs = self.dc.forward(sqdist, split)
        return probs



    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)] 
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'], split)
        neg_scores = self.decode(embeddings, edges_false, split)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

