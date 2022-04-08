# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from fblib.util.helpers import worker_seed
import fblib.util.pdf_visualizer as viz

# Losses
from fblib.layers.loss import BalancedCrossEntropyLoss, SoftMaxwithLoss, NormalsLoss, DepthLoss

# Dataloaders
import fblib.dataloaders as dbs
from fblib.dataloaders.combine_im_dbs import CombineIMDBs
from fblib.layers.loss import normal_ize

# Transformations
from fblib.dataloaders import custom_transforms as tr

# Collate for MIL
from fblib.util.custom_collate import collate_mil


def accuracy(output, target, topk=(1,), ignore_label=255):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = (target != ignore_label).sum().item()
    if batch_size == 0:
        return -1

    _, pred = output.topk(maxk, 1, True, True)
    if pred.shape[-1] == 1:
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    else:
        correct = pred.eq(target.unsqueeze(1))

    res = []
    for _ in topk:
        correct_k = correct[:].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def eval_all_results(p):
    if 'DO_SEMSEG' in p and p.DO_SEMSEG:
        from fblib.evaluation.eval_semseg import eval_and_store_semseg
        for db in p['infer_db_names']:
            eval_and_store_semseg(database=db,
                                  save_dir=p['save_dir_root'],
                                  exp_name=p['exp_name'],
                                  overfit=p.overfit)

    if 'DO_HUMAN_PARTS' in p and p.DO_HUMAN_PARTS:
        from fblib.evaluation.eval_human_parts import eval_and_store_human_parts
        for db in p['infer_db_names']:
            eval_and_store_human_parts(database=db,
                                       save_dir=p['save_dir_root'],
                                       exp_name=p['exp_name'],
                                       overfit=p.overfit)

    if 'DO_NORMALS' in p and p.DO_NORMALS:
        from fblib.evaluation.eval_normals import eval_and_store_normals
        for db in p['infer_db_names']:
            eval_and_store_normals(database=db,
                                   save_dir=p['save_dir_root'],
                                   exp_name=p['exp_name'],
                                   overfit=p.overfit)

    if 'DO_SAL' in p and p.DO_SAL:
        from fblib.evaluation.eval_sal import eval_and_store_sal
        for db in p['infer_db_names']:
            eval_and_store_sal(database=db,
                               save_dir=p['save_dir_root'],
                               exp_name=p['exp_name'],
                               overfit=p.overfit)

    if 'DO_DEPTH' in p and p.DO_DEPTH:
        from fblib.evaluation.eval_depth import eval_and_store_depth
        for db in p['infer_db_names']:
            eval_and_store_depth(database=db,
                                 save_dir=p['save_dir_root'],
                                 exp_name=p['exp_name'],
                                 overfit=p.overfit)

    if 'DO_ALBEDO' in p and p.DO_ALBEDO:
        from fblib.evaluation.eval_albedo import eval_and_store_albedo
        for db in p['infer_db_names']:
            eval_and_store_albedo(database=db,
                                  save_dir=p['save_dir_root'],
                                  exp_name=p['exp_name'],
                                  overfit=p.overfit)

    if 'DO_EDGE' in p and p.DO_EDGE and p['eval_edge']:
        from fblib.evaluation.eval_edges import sync_and_evaluate_one_folder
        for db in p['infer_db_names']:
            sync_and_evaluate_one_folder(database=db,
                                         save_dir=p['save_dir_root'],
                                         exp_name=p['exp_name'],
                                         prefix=p['tasks_name'],
                                         all_tasks_present=(p.MINI if 'MINI' in p else False))


def get_transformations(p, flag_old_totensor=False):
    """
    Get the transformations for training and testing
    """

    # Training transformations

    # Horizontal flips with probability of 0.5
    transforms_tr = [tr.RandomHorizontalFlip()]

    # Rotations and scaling
    transforms_tr.extend([tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25),
                                          flagvals={x: p.TASKS.FLAGVALS[x] for x in p.TASKS.FLAGVALS})])
    # Fixed Resize to input resolution
    transforms_tr.extend([tr.FixedResize(resolutions={x: tuple(p.TRAIN.SCALE) for x in p.TASKS.FLAGVALS},
                                         flagvals={x: p.TASKS.FLAGVALS[x] for x in p.TASKS.FLAGVALS})])

    if flag_old_totensor:
        transforms_tr.extend([tr.AddIgnoreRegions(), tr.ToTensor_old()])
    else:
        transforms_tr.extend([tr.AddIgnoreRegions(), tr.ToTensor()])
    transforms_tr = transforms.Compose(transforms_tr)

    # Testing (during training transforms)
    transforms_ts = []
    transforms_ts.extend([tr.FixedResize(resolutions={x: tuple(p.TEST.SCALE) for x in p.TASKS.FLAGVALS},
                                         flagvals={x: p.TASKS.FLAGVALS[x] for x in p.TASKS.FLAGVALS})])
    if flag_old_totensor:
        transforms_ts.extend([tr.AddIgnoreRegions(), tr.ToTensor_old()])
    else:
        transforms_ts.extend([tr.AddIgnoreRegions(), tr.ToTensor()])
    transforms_ts = transforms.Compose(transforms_ts)

    # Transformations to be used during inference
    transforms_infer = transforms_ts

    return transforms_tr, transforms_ts, transforms_infer


def get_loss(p, task=None):
    if task == 'edge':
        criterion = BalancedCrossEntropyLoss(size_average=True, pos_weight=p['edge_w'])
    elif task == 'semseg' or task == 'human_parts':
        criterion = SoftMaxwithLoss()
    elif task == 'normals':
        criterion = NormalsLoss(normalize=True, size_average=True, norm=p['normloss'])
    elif task == 'sal':
        criterion = BalancedCrossEntropyLoss(size_average=True)
    elif task == 'depth':
        criterion = DepthLoss()
    elif task == 'albedo':
        criterion = torch.nn.L1Loss(reduction='elementwise_mean')
    else:
        raise NotImplementedError('Undefined Loss: Choose a task among '
                                  'edge, semseg, human_parts, sal, depth, albedo, or normals')

    return criterion


def get_train_loader(p, db_name, transforms):
    print('Preparing train loader for db: {}'.format(db_name))

    db_names = [db_name] if isinstance(db_name, str) else db_name
    dbs_train = {}

    for db in db_names:
        if db == 'PASCALContext':
            dbs_train[db] = dbs.PASCALContext(split=['train'], transform=transforms, retname=True,
                                              do_edge=p.DO_EDGE, do_human_parts=p.DO_HUMAN_PARTS,
                                              do_semseg=p.DO_SEMSEG, do_normals=p.DO_NORMALS, do_sal=p.DO_SAL,
                                              overfit=p['overfit'])
        elif db == 'VOC12':
            dbs_train[db] = dbs.VOC12(split=['train'], transform=transforms, retname=True,
                                      do_semseg=p.DO_SEMSEG, overfit=p['overfit'])
        elif db == 'SBD':
            dbs_train[db] = dbs.SBD(split=['train', 'val'], transform=transforms, retname=True,
                                    do_semseg=p.DO_SEMSEG, overfit=p['overfit'])
        elif db == 'NYUD_nrm':
            dbs_train[db] = dbs.NYUDRaw(split='train', transform=transforms, overfit=p['overfit'])
        elif db == 'NYUD':
            dbs_train[db] = dbs.NYUD_MT(split='train', transform=transforms, do_edge=p.DO_EDGE, do_semseg=p.DO_SEMSEG,
                                        do_normals=p.DO_NORMALS, do_depth=p.DO_DEPTH, overfit=p['overfit'])
        elif db == 'COCO':
            dbs_train[db] = dbs.COCOSegmentation(split='train2017', transform=transforms, retname=True,
                                                 area_range=[1000, float("inf")], only_pascal_categories=True,
                                                 overfit=p['overfit'])
        elif db == 'FSV':
            dbs_train[db] = dbs.FSVGTA(split='train', mini=False, transform=transforms, retname=True,
                                       do_semseg=p.DO_SEMSEG, do_albedo=p.DO_ALBEDO, do_depth=p.DO_DEPTH,
                                       overfit=p['overfit'])
        else:
            raise NotImplemented("train_db_name: Choose among BSDS500, PASCALContext, VOC12, COCO, FSV, and NYUD")

    if len(dbs_train) == 1:
        db_train = dbs_train[list(dbs_train.keys())[0]]
    else:
        db_exclude = dbs.VOC12(split=['val'], transform=transforms, retname=True,
                               do_semseg=p.DO_SEMSEG, overfit=p['overfit'])
        db_train = CombineIMDBs([dbs_train[x] for x in dbs_train], excluded=[db_exclude], repeat=[1, 1])

    trainloader = DataLoader(db_train, batch_size=p['trBatch'], shuffle=True, drop_last=True,
                             num_workers=4, worker_init_fn=worker_seed, collate_fn=collate_mil)
    return trainloader


def get_test_loader(p, db_name, transforms, infer=False):
    print('Preparing test loader for db: {}'.format(db_name))

    if db_name == 'BSDS500':
        db_test = dbs.BSDS500(split=['test'], transform=transforms, overfit=p['overfit'])
    elif db_name == 'PASCALContext':
        db_test = dbs.PASCALContext(split=['val'], transform=transforms,
                                    retname=True, do_edge=p.DO_EDGE, do_human_parts=p.DO_HUMAN_PARTS,
                                    do_semseg=p.DO_SEMSEG, do_normals=p.DO_NORMALS, do_sal=p.DO_SAL,
                                    overfit=p['overfit'])
    elif db_name == 'VOC12':
        db_test = dbs.VOC12(split=['val'], transform=transforms,
                            retname=True, do_semseg=p.DO_SEMSEG, overfit=p['overfit'])
    elif db_name == 'NYUD':
        db_test = dbs.NYUD_MT(split='val', transform=transforms, do_edge=p.DO_EDGE, do_semseg=p.DO_SEMSEG,
                              do_normals=p.DO_NORMALS, do_depth=p.DO_DEPTH, overfit=p['overfit'])
    elif db_name == 'COCO':
        db_test = dbs.COCOSegmentation(split='val2017', transform=transforms, retname=True,
                                       area_range=[1000, float("inf")], only_pascal_categories=True,
                                       overfit=p['overfit'])
    elif db_name == 'FSV':
        db_test = dbs.FSVGTA(split='test', mini=True, transform=transforms, retname=True,
                             do_semseg=p.DO_SEMSEG, do_albedo=p.DO_ALBEDO, do_depth=p.DO_DEPTH,
                             overfit=p['overfit'])
    else:
        raise NotImplemented("test_db_name: Choose among BSDS500, PASCALContext, VOC12, COCO, FSV, and NYUD")

    drop_last = False if infer else True
    testloader = DataLoader(db_test, batch_size=p.TEST.BATCH_SIZE, shuffle=False, drop_last=drop_last,
                            num_workers=2, worker_init_fn=worker_seed)

    return testloader


def get_output(output, task):
    output = output.permute(0, 2, 3, 1)
    if task == 'normals':
        output = (normal_ize(output, dim=3) + 1.0) * 255 / 2.0
    elif task in {'semseg', 'human_parts'}:
        _, output = torch.max(output, dim=3)
    elif task in {'edge', 'sal'}:
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)))
    elif task in {'depth'}:
        pass
    else:
        raise ValueError('Select one of the valid tasks')

    return output.cpu().data.numpy()
