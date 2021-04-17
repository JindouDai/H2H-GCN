A Hyperbolic-to-Hyperbolic Graph Convolutional Network (H2H-GCN)
======================================================

This repository includes the implementations of the proposed H2H-GCN for the link prediction and node classification tasks on the Disease dateset [1] in PyTorch. 

Before running the model, please create environment according to "requirments.txt".

### For link prediction, run
'''
python train.py --task lp --dataset disease_lp --model H2HGCN  --normalize-feats 0 --log-freq 20   --epochs 8000  --step_lr_reduce_freq 7000 --feature_dim 11  --tie_weight True --patience 1000  --lr 0.001 --lr_stie 0.001  --dim 16 --num-layers 2
'''

### For node classification, run
'''
python train.py --task nc --dataset disease_nc --model H2HGCN --log-freq 20  --lr_scheduler step --epochs 5000 --step_lr_reduce_freq 5000 --feature_dim 1000  --tie_weight True   --lr 0.01  --lr_stie 0.01 --num_centroid 200 --dim 16  --num-layers 5
'''

optional arguments:
    --task                  which tasks to train on, 'lp' or 'nc'  
    --dataset               which dataset to use, 'disease_lp' or 'disease_nc'
    --model                 which model to use, 'H2HGCN' or 'HGCN'
    --lr                    learning rate for Euclidean parameters
    --lr_stie               learning rate for the Stiefel parameters
    --normalize-feats       whether to normalize input node features
    --epochs                maximum number of epochs
    --step_lr_reduce_freq   step_size for StepLR scheduler 
    --feature_dim           feature_dim input feature dimensionality
    --dim                   embedding dimensionality
    --num-layers            number of hidden layers
    --patience              patience for early stopping
    --num_centroid          number of centroids used for the node classification task

Directory
   data                     dataset files, including the "disease_lp" and "disease_nc"
   layers                   include a centroid-based classification and layers used in H2H-GCN
   log                      path to save logs
   manifolds                include the Lorentz manifold and the Stiefel manifold
   model_save               path to save trained models
   models                   encoder for graph embedding and decoder for post-processing
   optimizers               optimizers for orthogonal parameters
   utils                    utility modules and functions
   config.py                config file
   train.py                 run this file to start the training
   requirements.txt         requirements file
   README.md                README file


### References
[1] [Chami, I., Ying, R., Ré, C. and Leskovec, J. Hyperbolic Graph Convolutional Neural Networks. NIPS 2019.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7108814/)
