import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.01, 'learning rate'),
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (5000, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'optimizer': ('Adam', 'which optimizer to use'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (100, 'patience for early stopping'),
        'seed': (1234, 'seed for training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'test-freq': (1, 'how often to compute test metrics (in epochs)'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs')
    },
    'model_config': {
        'task': ('lp', 'which tasks to train on, can be any of [lp, nc]'),
        'model': ('H2HGCN', 'our model'),
        'dim': (128, 'embedding dimension'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'double-precision': ('0', 'whether to use double precision')
    },
    'data_config': {
        'dataset': ('disease_lp', 'which dataset to use'),
        'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'use-feats': (1, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split-seed': (1234, 'seed for data splits (train/test/val)'),
    },
    'my_config': {
        'lr_stie': (0.1, 'learning lare for Stiefel paramsters'),
        'stie_vars': ([], 'stiefel parameters'),
        'eucl_vars': ([], 'Euclidean parameters'),
        'stiefel_optimizer': ('rsgd', 'optimizer for stiefel parameters'),
        'lr_scheduler': ('step', 'which scheduler to use'),
        'lr_gamma': (0.98, 'gamma for scheduler'),
        'step_lr_gamma': (0.1, 'gamma for StepLR scheduler'),
        'step_lr_reduce_freq': (500, 'step size for StepLR scheduler'),
        'weight_decay': (0.0, 'weight decay'),
        'proj_init': ('xavier', 'the way to initialize parameters'),
        'embed_manifold': ('euclidean', ''),
        'tie_weight': (True, 'whether to tie transformation matrices'),
        'select_manifold': ('lorentz', 'selected manifold'),
        "num_centroid": (200, 'number of centroids'),
        "weight_manifold": ("StiefelManifold", 'Stiefel parameters'),
        "feature_dim": (1, "input feature dimensionality",),
        "pre_trained":(False, "whether use pre-train model"),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)