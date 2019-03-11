"""
=================================================
                    MAIN FILE
            FOR SETTING HYPER PARAMETERS
=================================================
"""


import loader
import torch
import os
import argparse
import network
import training
from argparse import RawTextHelpFormatter
import numpy as np
import valid

"""
=================================================
    HYPER PARAMETERS, CONSTANTS, NETWORK
=================================================
"""


parser = argparse.ArgumentParser(description='test', formatter_class=RawTextHelpFormatter)

parser.add_argument("-sig", action="store", dest="sigma", default=0.01, type=float)
parser.add_argument("-img", action="store", dest="image_size", default=64, type=int)
parser.add_argument("-N", action="store", dest="N", default=5, type=int)
parser.add_argument("-T", action="store", dest="T", default=10, type=int)
parser.add_argument("-epochs1", action="store", dest="epochs1", default=5, type=int)
parser.add_argument("-epochs2", action="store", dest="epochs2", default=5, type=int)
parser.add_argument("-dim", action="store", dest="dim", default=100, type=int)
parser.add_argument("-lr", action="store", dest="lr", default=0.0001, type=float)
parser.add_argument("-save_every", action="store", dest="save_every", default=5, type=int)
parser.add_argument("-vid_dir", action="store", dest="vid_dir", default='.', type=str)


args = parser.parse_args()

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")


"""
=================================================
    PREPARATION AND PREREQUISITES FOR RUN-UP
=================================================
"""

# load training/test set
with open('train_set.txt') as f:
    training_set_titles = f.read().splitlines()

with open('valid_set.txt') as f:
    validating_set_titles = f.read().splitlines()


# create net

net = network.CNN_LSTM(in_image_dim=args.image_size,
                       characteristic_dim=args.dim)

# choose optimizer
optimizer = torch.optim.Adam(net.parameters(),
                             lr=args.lr)


# GPU
use_gpu = torch.cuda.is_available()

if use_gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    net = net.cuda()
    print('USE GPU')
else:
    print('USE CPU')


print("LOAD DATA VIDEOS...")
# prepare train and test sets
training_set_videos = loader.load_videos(training_set_titles, args.image_size, use_gpu, 'train', args.vid_dir)
validating_set_videos = loader.load_videos(validating_set_titles, args.image_size, use_gpu, 'valid', args.vid_dir)
print("END LOADING!", end="\n"*2)

"""
=================================================
                TRAINING PHASE 
=================================================
"""

net = training.train(training_set_videos=training_set_videos,
                     net=net, optimizer=optimizer,
                     save_every=args.save_every,
                     T=args.T, N=args.N, sigma=args.sigma,
                     epochs1=args.epochs1, epochs2=args.epochs2,
                     use_gpu=use_gpu)

"""
=================================================
                Valid PHASE 
=================================================
"""

valid.compute_tests(validating_set_videos, net, pad_len=max(5, args.T), use_gpu=use_gpu)


