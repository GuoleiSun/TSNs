# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import cv2
import argparse
import torch
import tarfile
from six.moves import urllib
from easydict import EasyDict as edict

# Networks
import fblib.networks.deeplab_multi_task.deeplab_se_resnet_multitask as se_resnet_multitask

# Common configs
from experiments.dense_predict.common_configs import get_loss, get_train_loader, get_test_loader, get_transformations

from fblib.util.mypath import Path


# def parse_args():

#     def str2bool(v):
#         return v.lower() in ("yes", "true", "t", "1")

#     parser = argparse.ArgumentParser(description='Multi-task Learning for NYUD')

#     # Select tasks
#     parser.add_argument('--active_tasks', type=int, nargs='+', default=[1, 1, 1, 1],
#                         help='Which tasks to train?')

#     # General parameters
#     parser.add_argument('--arch', type=str, default='se_res50',
#                         help='network: se_res26, se_res50, se_res101')
#     parser.add_argument('--pretr', type=str, default='imagenet',
#                         help='pre-trained model: "imagenet" or "scratch"')
#     parser.add_argument('--trBatch', type=int, default=8,
#                         help='training batch size')
#     parser.add_argument('-lr', type=float, default=0.001,
#                         help='initial learning rate. poly-learning rate is used.')
#     parser.add_argument('--lr_dec', type=float, default=1,
#                         help='decoder learning rate multiplier')
#     parser.add_argument('-wd', type=float, default=1e-4,
#                         help='Weight decay')
#     parser.add_argument('--epochs', type=int, default=200,
#                         help='Total number of epochs for training')
#     parser.add_argument('--resume_epoch', type=int, default=0,
#                         help='Resume Epoch #')
#     parser.add_argument('--cls', type=str, default='atrous-v3',
#                         help='Classifier type')
#     parser.add_argument('--stride', type=int, default=16,
#                         help='Output stride of ResNet backbone. If set to 16 saves significant memory')
#     parser.add_argument('--trNorm', type=str2bool, default=True,
#                         help='train normalization layers of Backbone?')
#     parser.add_argument('--dec_w', type=int, default=64,
#                         help='decoder width (default 256 in Deeplab v3+')
#     parser.add_argument('--overfit', type=str2bool, default=False,
#                         help='overfit to small subset of data for debugging purposes')

#     # Modulation Parameters
#     parser.add_argument('--seenc', type=str2bool, default=True,
#                         help='Squeeze and excitation per task for encoder? False will still use 1 SE for all tasks')
#     parser.add_argument('--sedec', type=str2bool, default=True,
#                         help='Squeeze and excitation per task for decoder? False will not use SE modules')
#     parser.add_argument('--adapt', type=str2bool, default=True,
#                         help='Use parallel residual adapters?')
#     parser.add_argument('--lr_tsk', type=float, default=-1,
#                         help='Task Specific layer learning rate multiplier')

#     # Discriminator parameters
#     parser.add_argument('--dscr', type=str, default='fconv',
#                         help='Use discriminator?')
#     parser.add_argument('--lr_dscr', type=int, default=10,
#                         help='learning rate multiplier of discriminator?')
#     parser.add_argument('--dscr_w', type=float, default=0.01,
#                         help='weight of discriminator in the range [0, 1]')
#     parser.add_argument('--dscrd', type=int, default=2,
#                         help='Depth of discriminator')
#     parser.add_argument('--dscrk', type=int, default=3,
#                         help='Kernel size of discriminator')

#     # Task-specific parameters
#     parser.add_argument('--edge_w', type=float, default=0.8,
#                         help='Weighting the positive loss for boundary detection as w * L_pos + (1 - w) * L_neg')

#     return parser.parse_args()

def parse_args_not_read_arguments():
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    class any_class:
        pass

    args=any_class()
    # parser = argparse.ArgumentParser(description='Multi-task learning with PASCAL')

    # Select tasks
    # parser.add_argument('--active_tasks', type=int, nargs='+', default=[1, 1, 1, 1, 1],
    #                     help='Which tasks to train?')
    args.active_tasks=[1, 1, 1, 1]
    # parser.add_argument('--onlyVOC', type=str2bool, default=False,
    #                     help='Use standard training and testing for semantic segmentation')
    # args.onlyVOC=False
    # General parameters
    # parser.add_argument('--arch', type=str, default='se_res26',
    #                     help='network: se_res26, se_res50, se_res101')
    args.arch='se_res50'
    # parser.add_argument('--pretr', type=str, default='imagenet',
    #                     help='pre-trained model: "imagenet" or "scratch"')
    args.pretr='imagenet'
    # parser.add_argument('--trBatch', type=int, default=8,
    #                     help='training batch size')
    args.trBatch=8
    # parser.add_argument('-lr', type=float, default=0.001,
    #                     help='initial learning rate. poly-learning rate is used.')
    args.lr=0.001
    # parser.add_argument('--lr_dec', type=float, default=1,
    #                     help='decoder learning rate multiplier')
    args.lr_dec=1
    # parser.add_argument('-wd', type=float, default=1e-4,
    #                     help='Weight decay')
    args.wd=1e-4
    # parser.add_argument('--epochs', type=int, default=60,
    #                     help='Total number of epochs for training')
    args.epochs=200
    # parser.add_argument('--resume_epoch', type=int, default=0,
    #                     help='Resume Epoch #')
    args.resume_epoch=0
    # parser.add_argument('--cls', type=str, default='atrous-v3',
    #                     help='Classifier type')
    args.cls='atrous-v3'
    # parser.add_argument('--stride', type=int, default=16,
    #                     help='Output stride of ResNet backbone. If set to 16 saves significant memory')
    args.stride=16
    # parser.add_argument('--trNorm', type=str2bool, default=True,
    #                     help='train normalization layers of Backbone?')
    args.trNorm=True
    # parser.add_argument('--dec_w', type=int, default=64,
    #                     help='decoder width (default 256 in Deeplab v3+')
    args.dec_w=64
    # parser.add_argument('--overfit', type=str2bool, default=False,
    #                     help='overfit to small subset of data for debugging purposes')
    args.overfit=False

    # Modulation Parameters
    # parser.add_argument('--seenc', type=str2bool, default=True,
    #                     help='Squeeze and excitation per task for encoder? False will still use 1 SE for all tasks')
    args.seenc=True
    # parser.add_argument('--sedec', type=str2bool, default=True,
    #                     help='Squeeze and excitation per task for decoder? False will not use SE modules')
    args.sedec=True
    # parser.add_argument('--adapt', type=str2bool, default=True,
    #                     help='Use parallel residual adapters?')
    args.adapt=True
    # parser.add_argument('--lr_tsk', type=float, default=-1,
    #                     help='Task Specific layer learning rate multiplier')
    args.lr_tsk=-1

    # Discriminator parameters
    # parser.add_argument('--dscr', type=str, default='fconv',
    #                     help='Use discriminator?')
    args.dscr='fconv'
    # parser.add_argument('--lr_dscr', type=int, default=10,
    #                     help='learning rate multiplier of discriminator?')
    args.lr_dscr=10
    # parser.add_argument('--dscr_w', type=float, default=0.01,
    #                     help='weight of discriminator in the range [0, 1]')
    args.dscr_w=0.01
    # parser.add_argument('--dscrd', type=int, default=2,
    #                     help='Depth of discriminator')
    args.dscrd=2
    # parser.add_argument('--dscrk', type=int, default=1,
    #                     help='kernel size of discriminator')
    args.dscrk=3

    # Task-specific parameters
    # parser.add_argument('--edge_w', type=float, default=0.95,
    #                     help='weighting the positive loss for boundary detection as w * L_pos + (1 - w) * L_neg')
    args.edge_w=0.8

    return args

def create_config():

    cfg = edict()

    # args = parse_args()
    args = parse_args_not_read_arguments()

    # Parse tasks
    assert (len(args.active_tasks) == 4)
    args.do_edge = args.active_tasks[0]
    args.do_semseg = args.active_tasks[1]
    args.do_normals = args.active_tasks[2]
    args.do_depth = args.active_tasks[3]

    print('\nThis script was run with the following parameters:')
    for x in vars(args):
        print('{}: {}'.format(x, str(getattr(args, x))))

    cfg.resume_epoch = args.resume_epoch

    cfg.DO_EDGE = args.do_edge
    cfg.DO_SEMSEG = args.do_semseg
    cfg.DO_NORMALS = args.do_normals
    cfg.DO_DEPTH = args.do_depth

    if not cfg.DO_EDGE and not cfg.DO_SEMSEG and not cfg.DO_NORMALS and not cfg.DO_DEPTH:
        raise ValueError("Select at least one task")

    cfg['arch'] = args.arch
    cfg['pretr'] = args.pretr
    cfg['trBatch'] = args.trBatch
    cfg['lr'] = args.lr
    cfg['lr_dec'] = args.lr_dec
    cfg['wd'] = args.wd
    cfg['cls'] = args.cls
    cfg['epochs'] = args.epochs
    cfg['stride'] = args.stride
    cfg['trNorm'] = args.trNorm
    cfg['dec_w'] = args.dec_w

    # Set Modulation (Squeeze and Exciation, Residual Adapters) parameters
    cfg['seenc'] = args.seenc
    cfg['sedec'] = args.sedec
    cfg['adapters'] = args.adapt

    if cfg['sedec']:
        cfg['norm_per_task'] = True
    else:
        cfg['norm_per_task'] = False

    if args.dscr == 'None':
        args.dscr = None

    cfg['dscr_type'] = args.dscr
    cfg['lr_dscr'] = args.lr_dscr
    cfg['dscr_w'] = args.dscr_w
    cfg['dscrd'] = args.dscrd
    cfg['dscrk'] = args.dscrk

    task_args, name_args = get_exp_name(args)

    cfg['exp_folder_name'] = 'nyud_resnet'
    cfg['exp_name'] = "_".join(name_args)
    cfg['tasks_name'] = "_".join(task_args)
    cfg['save_dir_root'] = os.path.join(Path.exp_dir(), cfg['exp_folder_name'], cfg['tasks_name'])
    cfg['train_db_name'] = ['NYUD', ]
    cfg['test_db_name'] = 'NYUD'
    cfg['infer_db_names'] = ['NYUD', ]

    # Which tasks?
    cfg.TASKS = edict()
    cfg.TASKS.NAMES = []
    cfg.TASKS.NUM_OUTPUT = {}  # How many outputs per task?
    cfg.TASKS.TB_MIN = {}
    cfg.TASKS.TB_MAX = {}
    cfg.TASKS.LOSS_MULT = {}
    cfg.TASKS.FLAGVALS = {'image': cv2.INTER_CUBIC}
    cfg.TASKS.INFER_FLAGVALS = {}

    if cfg.DO_EDGE:
        # Edge Detection
        print('Adding task: Edge Detection')
        tmp = 'edge'
        cfg.TASKS.NAMES.append(tmp)
        cfg.TASKS.NUM_OUTPUT[tmp] = 1
        cfg.TASKS.TB_MIN[tmp] = 0
        cfg.TASKS.TB_MAX[tmp] = cfg.TASKS.NUM_OUTPUT[tmp]
        cfg.TASKS.LOSS_MULT[tmp] = 50
        cfg.TASKS.FLAGVALS[tmp] = cv2.INTER_NEAREST
        cfg.TASKS.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR

        # Add task-specific parameters from parser
        cfg['edge_w'] = args.edge_w
        cfg['eval_edge'] = False

    if cfg.DO_SEMSEG:
        # Semantic Segmentation
        print('Adding task: Semantic Segmentation')
        tmp = 'semseg'
        cfg.TASKS.NAMES.append(tmp)
        cfg.TASKS.NUM_OUTPUT[tmp] = 41
        cfg.TASKS.TB_MIN[tmp] = 0
        cfg.TASKS.TB_MAX[tmp] = cfg.TASKS.NUM_OUTPUT[tmp] - 1
        cfg.TASKS.LOSS_MULT[tmp] = 1
        cfg.TASKS.FLAGVALS[tmp] = cv2.INTER_NEAREST
        cfg.TASKS.INFER_FLAGVALS[tmp] = cv2.INTER_NEAREST

    if cfg.DO_NORMALS:
        # Human Parts Segmentation
        print('Adding task: Normals')
        tmp = 'normals'
        cfg.TASKS.NAMES.append(tmp)
        cfg.TASKS.NUM_OUTPUT[tmp] = 3
        cfg.TASKS.TB_MIN[tmp] = -1
        cfg.TASKS.TB_MAX[tmp] = 1
        cfg.TASKS.LOSS_MULT[tmp] = 10
        cfg.TASKS.FLAGVALS[tmp] = cv2.INTER_CUBIC
        cfg.TASKS.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR

        cfg['normloss'] = 1  # Hard-coded L1 loss for normals

    if cfg.DO_DEPTH:
        # Depth Estimation
        print('Adding task: Depth')
        tmp = 'depth'
        cfg.TASKS.NAMES.append(tmp)
        cfg.TASKS.NUM_OUTPUT[tmp] = 1
        cfg.TASKS.TB_MIN[tmp] = 1
        cfg.TASKS.TB_MAX[tmp] = 10
        cfg.TASKS.LOSS_MULT[tmp] = 1
        cfg.TASKS.FLAGVALS[tmp] = cv2.INTER_LINEAR
        cfg.TASKS.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR

    cfg['lr_tsk'] = len(cfg.TASKS.NAMES) if args.lr_tsk < 0 else args.lr_tsk

    cfg.NETWORK = edict()

    # Visualize the network on Tensorboard / pdf?
    cfg.NETWORK.VIS_NET = False

    cfg.TRAIN = edict()
    cfg.TRAIN.SCALE = (512, 512)
    # cfg.TRAIN.SCALE = (1024, 1024)
    # cfg.TRAIN.SCALE = (560, 560)
    # cfg.TRAIN.SCALE = (256, 256)
    cfg.TRAIN.MOMENTUM = 0.9
    cfg.TRAIN.TENS_VIS = True
    cfg.TRAIN.TENS_VIS_INTER = 1000
    cfg.TRAIN.TEMP_LOSS_INTER = 1000

    cfg.TEST = edict()

    # See evolution of the test set when training?
    cfg.TEST.USE_TEST = True
    cfg.TEST.TEST_INTER = 10
    cfg.TEST.SCALE = (512, 512)
    # cfg.TEST.SCALE = (1024, 1024)
    # cfg.TEST.SCALE = (560, 560)
    # cfg.TEST.SCALE = (256, 256)

    cfg.SEED = 0
    cfg.EVALUATE = True
    cfg.DEBUG = False

    cfg['overfit'] = args.overfit
    if cfg['overfit']:
        cfg['save_dir_root'] = os.path.join(Path.exp_dir(), cfg['exp_folder_name'])
        cfg['exp_name'] = 'test'

    cfg['save_dir'] = os.path.join(cfg['save_dir_root'], cfg['exp_name'])
    return cfg


def check_downloaded(p):
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> %s %.1f%%' %
                         (_fpath, float(count * block_size) /
                          float(total_size) * 100.0))
        sys.stdout.flush()

    def _create_url(name):
        return 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL//models/astmt/{}.tgz'.format(name)

    _model_urls = {
        'nyud_resnet_edge_semseg_normals_depth_'
        'arch-se_res50_pretr-imagenet_trBatch-8_lr-0.001_epochs-200_trNorm_seenc_sedec_adapt_dscr-fconv_lr_dscr'
        '-10_dscr_w-0.01_dscrd-2_dscrk-3_edge_w-0.8_200',
    }

    ans = False
    _check = p['exp_folder_name'] + '_' + p['tasks_name'] + '_' + p['exp_name'] + '_' + str(p['resume_epoch'])
    _fpath = os.path.join(Path.exp_dir(), _check + '.tgz')

    if _check in _model_urls:
        if not os.path.isfile(os.path.join(p['save_dir'], 'models',
                                           'model_epoch-' + str(p['resume_epoch'] - 1) + '.pth')):
            urllib.request.urlretrieve(_create_url(_check), _fpath, _progress)

            # extract file
            cwd = os.getcwd()
            print('\nExtracting tar file')
            tar = tarfile.open(_fpath)
            os.chdir(Path.exp_dir())
            tar.extractall()
            tar.close()
            os.chdir(cwd)
            print('Done!')
        ans = True
    return ans


def get_net_resnet(p):
    """
    Define the network (standard Deeplab ResNet101) and the trainable parameters
    """

    if p['arch'] == 'se_res26':
        network = se_resnet_multitask.se_resnet26
    elif p['arch'] == 'se_res50':
        network = se_resnet_multitask.se_resnet50
    elif p['arch'] == 'se_res101':
        network = se_resnet_multitask.se_resnet101
    else:
        raise NotImplementedError('ResNet: Choose between among se_res26, se_res50, and se_res101')

    print('Creating ResNet model: {}'.format(p.NETWORK))

    net = network(tasks=p.TASKS.NAMES, n_classes=p.TASKS.NUM_OUTPUT, pretrained=p['pretr'], classifier=p['cls'],
                  output_stride=p['stride'], train_norm_layers=p['trNorm'], width_decoder=p['dec_w'],
                  squeeze_enc=p['seenc'], squeeze_dec=p['sedec'], adapters=p['adapters'],
                  norm_per_task=p['norm_per_task'], dscr_type=p['dscr_type'], dscr_d=p['dscrd'], dscr_k=p['dscrk'])

    if p['resume_epoch'] != 0:
        print("Initializing weights from: {}".format(
            os.path.join(p['save_dir'], 'models', 'model_epoch-' + str(p['resume_epoch'] - 1) + '.pth')))
        state_dict_checkpoint = torch.load(
            os.path.join(p['save_dir'], 'models', 'model_epoch-' + str(p['resume_epoch'] - 1) + '.pth')
            , map_location=lambda storage, loc: storage)

        net.load_state_dict(state_dict_checkpoint)

    return net


def get_train_params(net, p):

    train_params = [{'params': se_resnet_multitask.get_lr_params(net, part='backbone', tasks=p.TASKS.NAMES),
                     'lr': p['lr']},
                    {'params': se_resnet_multitask.get_lr_params(net, part='decoder',  tasks=p.TASKS.NAMES),
                     'lr': p['lr'] * p['lr_dec']},
                    {'params': se_resnet_multitask.get_lr_params(net, part='task_specific', tasks=p.TASKS.NAMES),
                     'lr': p['lr'] * p['lr_tsk']}]
    if p['dscr_type'] is not None:
        train_params.append(
            {'params': se_resnet_multitask.get_lr_params(net, part='discriminator', tasks=p.TASKS.NAMES),
             'lr': p['lr'] * p['lr_dscr']})

    return train_params


def get_exp_name(args):
    """
    Creates the name experiment from the configuration file and the arguments
    """
    task_dict = {
        'do_edge': 0,
        'do_semseg': 0,
        'do_normals': 0,
        'do_depth': 0
    }

    name_dict = {
        'arch': None,
        'pretr': None,
        'trBatch': None,
        'lr': None,
        'wd': 1e-04,
        'epochs': None,
        'cls': 'atrous-v3',
        'stride': 16,
        'trNorm': False,
        'dec_w': 64,
        'seenc': False,
        'sedec': False,
        'adapt': False,
        'dscr': None,
        'lr_dscr': 1,
        'dscr_w': ('dscr', None),
        'dscrd': ('dscr', None),
        'dscrk': ('dscr', None),
        'edge_w': ('do_edge', None),
        'lr_dec': 1,
        'lr_tsk': -1,
    }

    # Experiment folder (task) string
    task_args = [x.replace('do_', '') for x in task_dict if getattr(args, x) != task_dict[x]]

    # Experiment name string
    name_args = []
    for x in name_dict:

        # Check dependencies in tuple
        if isinstance(name_dict[x], tuple):
            elem = name_dict if name_dict[x][0] in name_dict else task_dict
            if elem[name_dict[x][0]] == getattr(args, name_dict[x][0]):
                continue

        if getattr(args, x) != name_dict[x]:
            tmp = getattr(args, x)
            if isinstance(tmp, list):
                tmp = "_".join([str(x) for x in tmp])
            else:
                tmp = str(tmp)
            name_args.append(x + '-' + tmp)
    name_args = [x.replace('-True', '') for x in name_args]

    return task_args, name_args
