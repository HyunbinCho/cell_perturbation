import torch
import pandas as pd
import numpy as np
import time
import os
import sys

from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import save_image

import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

from torchvision.models.resnet import resnet18
from optparse import OptionParser

from utils.load_dataset import *
from utils.logging_tensorboard import *

from create_batch_info import create_batch_info

import yaml


def train_net(net, merged_data, args, **kwargs):

    ##################### set parameters #######################
    step_size= 10 #learning rate decay step size
    reducelr_patience = 6
    lr_decay = 0.3

    data_path = kwargs.get('data_path', None)
    model_path = kwargs.get('model_path', None)
    #################################################################

    if not os.path.exists(os.path.join(model_path, args.model_index)):
        os.mkdir(os.path.join(model_path, args.model_index))

    ###optimizer###
    if args.opt == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    elif args.opt == "adam":
        # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise ValueError("you gave wrong a parameter for --optim : {}".format(args.opt))

    ###loss function###
    if args.loss == 'bce':
        print("loss : bce")
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'nll':
        print("loss : nll")
        criterion = nn.NLLLoss()
    elif args.loss == 'ce':
        print('loss : ce(Cross Entropy)')
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("you gave a wrong parameter for --loss : {}".format(args.loss))

    ###scheduler###
    if args.scheduler is not None:
        if args.scheduler == "steplr":
            scheduler = StepLR(optimizer=optimizer, step_size=step_size, gamma=lr_decay)
        elif args.scheduler == "reducelr":
            scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=reducelr_patience, factor=lr_decay)
        else:
            raise ValueError("you gave wrong parameters for --scheduler : {}".format(args.scheduler))

    val_benchmark = 10000000

    train_dataset = TrainDatasetRecursion(merged_data=merged_data, batch_info=batch_info_dict,
                                          args=args, isNormalize=args.normalize, isTrain=True, train_ratio=0.8, seed=10)
    val_dataset = TrainDatasetRecursion(merged_data=merged_data, batch_info=batch_info_dict,
                                        args=args, isNormalize=args.normalize, isTrain=False, train_ratio=0.8, seed=10)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    ####training & validation start#######
    for epoch in range(args.epochs):
        start_time = time.time()
        writer.add_text('Text', 'text logged at epoch: ' + str(epoch), epoch)  # temp for checking
        print('\n\nStarting epoch {}/{}.'.format(epoch + 1, args.epochs))
        writer_arguments(writer, args, epoch)

        epoch_train_loss = 0

        N_train = len(train_dataloader)
        net.train()
        for i, (img, labels) in enumerate(train_dataloader):
            #print(labels)
            img = img.cuda()
            labels = labels.cuda()

            outputs = net(img)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss = epoch_train_loss + loss.item()

        writer_training(writer, epoch_train_loss / N_train, epoch)
        print('Epoch finished! Train Loss: {}'.format(epoch_train_loss / N_train))

        # VALIDATION
        epoch_val_loss = eval_net(net, val_dataloader, criterion, args)
        writer_validation(writer, epoch_val_loss / len(val_dataloader), epoch)

        # If validation loss is lower than the previous model-checkpoint, a new model-checkpoint saved
        if val_benchmark > epoch_val_loss:
            val_benchmark = epoch_val_loss
            checkpoint_path = os.path.join(model_path, args.model_index, 'CP{}.pth'.format(epoch + 1))
            torch.save(net.state_dict(), checkpoint_path)
            print('Checkpoint {} saved'.format(epoch + 1))

        if args.scheduler == "steplr":
            scheduler.step()
        elif args.scheduler == "reducelr":
            scheduler.step(epoch_val_loss)

        taken_time = (time.time() - start_time)
        print('---> One Epoch Done. (', taken_time, 'sec)')


def eval_net(net, val_dataloader, criterion, args):
    epoch_val_loss = 0

    N_val = len(val_dataloader)
    net.eval()
    for i, (img, labels) in enumerate(val_dataloader):

        img = img.cuda()
        labels = labels.cuda()
        outputs = net(img)

        val_loss = criterion(outputs, labels)

        epoch_val_loss = epoch_val_loss + val_loss.item()

    print('Epoch finished!   Val Loss: {}'.format(epoch_val_loss / N_val))
    return epoch_val_loss



def get_args():
    parser = OptionParser()
    parser.add_option('-i', '--img-size', dest='img_size', default=512, type='int',
                      help='image size')
    parser.add_option('-l', '--resize', dest='resize', default=256, type='int',
                      help='resize ')
    parser.add_option('-e', '--epochs', dest='epochs', default=3, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=8,
                      type='int', help='batch size')
    parser.add_option('-r', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', dest='gpu', default=0,
                      type='int', help='what index of gpu will be used for this code')
    parser.add_option('-n', '--net', dest='net_name', type='str',
                      default="resnet18", help='network architecture')
    parser.add_option('-y', '--model-index', dest='model_index', type='str',
                      default="model_default", help='model_name (used to separate individual codes execution)')
    parser.add_option('-s', '--scheduler', dest='scheduler',
                      default=None, help="which scheduler(weight decay) is used")
    parser.add_option('--optim', dest='opt',
                      default='sgd', help="optimizer (currently available : sgd, adam)")
    parser.add_option('--loss', dest='loss',
                      default='ce',
                      help="loss function (currently available : ce)")
    parser.add_option('--normalize', dest='normalize', help="normalization")

    ####cell recursion-specific arguments
    parser.add_option('--cell', dest='cell',
                      default='all',
                      help='cell name to be used (if "all", all cells will be in dataset)')
    #parser.add_option('')

    (args, _) = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    ########### setting locations for kaggle ################
    data_path = "/kaggle/input/recursion-cellular-image-classification/"
    model_path = "/kaggle/working/model/"
    batch_info_path = "/kaggle/working/batch_info.yaml"
    metadata_path = "/kaggle/working/metadata.pickle"


    ########################################

    #### load dataset ####
    print("loading datasets :", time.ctime())

    print("creating batch info :", time.ctime())
    create_batch_info(data_path, batch_info_path, metadata_path)
    print("batch info created:", time.ctime())

    traindata = load_data_cell_perturbation(base_path=os.path.join(data_path, "train"))
    testdata = load_data_cell_perturbation(base_path=os.path.join(data_path, "test"))

    metadata = load_metadata(from_server=True, path=metadata_path)
    merged_data = merge_all_data_to_metadata([traindata, testdata], metadata)

    with open(batch_info_path, 'r', encoding="utf-8") as yaml_file:
        batch_info_dict = yaml.load(yaml_file)

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    print("loaded completely :", time.ctime())
    ##################################

    #############load parameters#######################
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    #create writer with tensorboardX
    if not os.path.exists('runs'):
        os.mkdir('runs')
    writer = SummaryWriter('runs/{}'.format(args.model_index))

    #### load network #####
    net = load_net(args.net_name)
    torch.backends.cudnn.benchmark = True
    net = net.cuda()

    train_net(net=net, merged_data=merged_data, args=args, model_path=model_path)





