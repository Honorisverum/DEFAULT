"""
=================================================
                    MAIN FILE
            FOR SETTING HYPER PARAMETERS
=================================================
"""


import loader
import torch
import metrics_plots
import os
import argparse
import network
import testing
import training
from argparse import RawTextHelpFormatter
import numpy as np

"""
=================================================
    HYPER PARAMETERS, CONSTANTS, NETWORK
=================================================
"""


parser = argparse.ArgumentParser(description='test', formatter_class=RawTextHelpFormatter)


parser.add_argument("-sig",
                    action="store",
                    dest="sigma",
                    default=0.001,
                    type=float,
                    help="sigma parameter\n"
                    )


parser.add_argument("-img",
                    action="store",
                    dest="image_size",
                    default=64,
                    type=int,
                    help="converted frame square side length\n"
                    )


parser.add_argument("-N",
                    action="store",
                    dest="N",
                    default=5,
                    type=int,
                    help="number of random samples made each epoch\n"
                    )


parser.add_argument("-T",
                    action="store",
                    dest="T",
                    default=10,
                    type=int,
                    help="processing sequence length\n"
                    )


parser.add_argument("-stages",
                    action="store",
                    dest="stages",
                    nargs=2,
                    default=[10, 10],
                    type=int,
                    help="a b    | initial and late stage length\n"
                    )


parser.add_argument("-dim",
                    action="store",
                    dest="dim",
                    default=100,
                    type=int,
                    help="characteristic model dimension\n"
                    )


parser.add_argument("-lr",
                    action="store",
                    dest="lr",
                    default=0.0001,
                    type=float,
                    help="learning rate\n"
                    )

parser.add_argument("-is_train",
                    action="store",
                    dest="is_train",
                    default=True,
                    type=bool,
                    help="is train?\n"
                    )


parser.add_argument("-save_every",
                    action="store",
                    dest="save_every",
                    default=5,
                    type=int,
                    help="saving period\n"
                    )


parser.add_argument("-save_file",
                    action="store",
                    dest="save_file",
                    default="last_weights.pt",
                    type=str,
                    help="save to file with .pt extension into weights folder\n"
                         "None if no need to save\n"
                    )


parser.add_argument("-load_file",
                    action="store",
                    dest="load_file",
                    default=None,
                    type=str,
                    help="load from file with .pt extension from weights folder\n"
                         "None if no need to load\n"
                    )


parser.add_argument("-vid_dir",
                    action="store",
                    dest="vid_dir",
                    default=None,
                    type=str,
                    help="None for don't change root dir \n"
                         "should be dir where all code and video folder \n"
                         "full directory, type ./.../ for get path deeply from root\n"
                    )


args = parser.parse_args()

SIGMA = args.sigma
IMAGE_SIZE = (args.image_size, args.image_size)
N = args.N
T = args.T
INITIAL_STAGE, LATE_STAGE = args.stages
MODEL_DIM = args.dim
LEARNING_RATE = args.lr
SAVE_EVERY = args.save_every
SAVE_FILENAME = args.save_file
LOAD_FILENAME = args.load_file
VID_DIR = args.vid_dir
IS_TRAIN = args.is_train


print("SIGMA:", SIGMA)
print("IMAGE SIZE:", IMAGE_SIZE)
print("N:", N)
print("T:", T)
print("INITIAL STAGE:", INITIAL_STAGE)
print("LATE_STAGE:", LATE_STAGE)
print("MODEL DIM:", MODEL_DIM)
print("LEARNING RATE:", LEARNING_RATE)
print("CHANGE DIR:", VID_DIR)
print("SAVE_EVERY:", SAVE_EVERY)
print("LOAD_FILENAME:", LOAD_FILENAME)
print("SAVE_FILENAME:", SAVE_FILENAME)
print("IS_TRAIN:", IS_TRAIN)


"""
=================================================
    PREPARATION AND PREREQUISITES FOR RUN-UP
=================================================
"""

# load training/test set
with open('train_set.txt') as f:
    training_set_titles = f.read().splitlines()

with open('test_set.txt') as f:
    testing_set_titles = f.read().splitlines()

EPOCHS = INITIAL_STAGE + LATE_STAGE

if LOAD_FILENAME == "None":
    LOAD_FILENAME = None

if VID_DIR == "None":
    VID_DIR = None

# create net
if LOAD_FILENAME is not None:
    load_path = os.path.join(os.getcwd(), 'weights', LOAD_FILENAME)
    net = torch.load(load_path, map_location='cpu')
else:
    net = network.CNN_LSTM(in_image_dim=IMAGE_SIZE,
                           characteristic_dim=MODEL_DIM)

# choose optimizer
optimizer = torch.optim.Adam(net.parameters(),
                             lr=LEARNING_RATE)


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
IS_TEST = (testing_set_titles != [])
IS_TRAIN = (training_set_titles != [])
training_set_videos = loader.load_videos(training_set_titles, T, IMAGE_SIZE, use_gpu, VID_DIR)
testing_set_videos = loader.load_videos(testing_set_titles, T, IMAGE_SIZE, use_gpu, VID_DIR)
print("END LOADING!", end="\n"*2)

"""
=================================================
                TRAINING PHASE 
=================================================
"""

if IS_TRAIN:
    weights_curve, weights_grad_curve,\
    rewards_curve, net = training.train(training_set_videos=training_set_videos,
                                        net=net, optimizer=optimizer,
                                        save_filename=SAVE_FILENAME,
                                        save_every=SAVE_EVERY,
                                        T=T, N=N, sigma=SIGMA,
                                        epochs=EPOCHS,
                                        initial_stage=INITIAL_STAGE,
                                        use_gpu=use_gpu)
else:
    weights_curve = np.loadtxt(os.getcwd() + "/results/weights.txt")
    weights_grad_curve = np.loadtxt(os.getcwd() + "/results/weights_grad.txt")
    rewards_curve = np.loadtxt(os.getcwd() + "/results/rewards.txt")

"""
=================================================
                TEST PHASE 
=================================================
"""


if IS_TEST:
    # forward pass on test videos
    testing.compute_tests(testing_set_videos, net, pad_len=max(5, T), use_gpu=use_gpu)

    # metrics
    metrics_plots.compute_metrics(testing_set_videos)

    # plots
    metrics_plots.draw_plots(rewards_curve, weights_curve, weights_grad_curve)
else:
    np.savetxt(os.getcwd() + "/results/rewards.txt", rewards_curve)
    np.savetxt(os.getcwd() + "/results/weights.txt", weights_curve)
    np.savetxt(os.getcwd() + "/results/weights_grad.txt", weights_grad_curve)

