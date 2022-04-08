### use "new split": the same split as kevis and menelaos's paper

from __future__ import print_function
from __future__ import division
import torch
import torchvision
print("PyTorch Version: ",torch.__version__)
#print("Torchvision Version: ",torchvision.__version__)
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import read_flags
from data_loader import VOCSegmentation, VOCSegmentation_new #, VOCSegmentation_new_fix_normal
from torchvision import transforms
from torch.utils.data import DataLoader
import PIL
from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image
import numpy as np
import random
import time
from torch import nn
from model import ResNetUNet, ResNetUNet2, ResNetUNet2_2, ResNetUNet2_2_no_adain
from torchsummary import summary
import albumentations as albu
import torch.nn.functional as F

## import kevis functions
from experiments.dense_predict.pascal_resnet import config as config
import fblib.dataloaders as dbs
from fblib.util.helpers import worker_seed
from fblib.util.custom_collate import collate_mil
from fblib.layers.loss import BalancedCrossEntropyLoss, SoftMaxwithLoss, NormalsLoss, DepthLoss
from fblib.util.model_resources.num_parameters import count_parameters



# IMG_SIZE = (3,256,256)
# IMG_SIZE = (3,512,512)
root_path = "/raid/guolei/fully-seg/PASCAL_MT"            #change this
checkpoint_path = ""

colourmap1 = {}
# colourmap['seg'] is the ground truth array for the semantic segmentation classes
colourmap = np.loadtxt('colourmap_21index.txt',delimiter='\t')
colourmap1['seg'] = np.zeros_like(colourmap)
colourmap1['seg'][:,0] = colourmap[:,0]
colourmap1['seg'][:,1] = colourmap[:,3]
colourmap1['seg'][:,2] = colourmap[:,2]
colourmap1['seg'][:,3] = colourmap[:,1]

# colourmap['parts'] is the ground truth array for the human parts segmentation classes
colourmap = np.loadtxt('colourmap7.txt',delimiter='\t')
colourmap1['parts'] = np.zeros_like(colourmap)
colourmap1['parts'][:,0] = colourmap[:,0]
colourmap1['parts'][:,1] = colourmap[:,3]
colourmap1['parts'][:,2] = colourmap[:,2]
colourmap1['parts'][:,3] = colourmap[:,1]

def obtain_dataloader_from_kevis(phases, task_sets, batch_size):
    # class EmptyParser():
    #     def parse_args():
    #         return
    # import argparse
    # argparse.ArgumentParser = EmptyParser
    

    p = config.create_config()
    transforms_tr, transforms_ts, _ = config.get_transformations(p)
    p['trBatch']=batch_size
    p.TEST.BATCH_SIZE = batch_size


    datasets = {}
    dataloaders = {}

    for phase in phases:
        datasets[phase] = {}
        dataloaders[phase] = {}

        for task in task_sets:
            p.DO_EDGE=0
            p.DO_HUMAN_PARTS=0
            p.DO_SEMSEG=0
            p.DO_NORMALS=0
            p.DO_SAL=0
            if task=='seg':
                p.DO_SEMSEG=1
            elif task=='parts':
                p.DO_HUMAN_PARTS=1
            elif task=='edges':
                p.DO_EDGE=1
            elif task=='normals':
                p.DO_NORMALS=1
            elif task=='saliency':
                p.DO_SAL=1

            if phase=='train':
                dataset, dataloader=get_train_loader(p, 'PASCALContext', transforms_tr)
                datasets[phase][task]=dataset
                dataloaders[phase][task]=dataloader
            elif phase=='val':
                dataset, dataloader=get_test_loader(p, 'PASCALContext', transforms_ts)
                datasets[phase][task]=dataset
                dataloaders[phase][task]=dataloader
    return datasets, dataloaders

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
    return db_train, trainloader

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

    return db_test, testloader

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)

def reduce_lr_poly(args, optimizer, global_iter, max_iter):
    base_lr = args.lr
    for g in optimizer.param_groups:
        g['lr'] = lr_poly(base_lr=base_lr, iter=global_iter, max_iter=max_iter, power=0.9)


class Poly_scheduler():

    def __init__(self, optimizer, max_epoch, momentum=0.9):
        self.optimizer = optimizer
        # self.global_step = 0
        self.max_epoch = max_epoch
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.optimizer.param_groups]


    def step(self, epoch):

        if epoch < self.max_epoch:
            lr_mult = (1 - float(self.epoch) / self.max_epoch) ** self.momentum

            for i in range(len(self.optimizer.param_groups)):
                self.optimizer.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult
        # super().step(closure)

        # self.global_step += 1


def main(flags):

    check_path = os.path.join(checkpoint_path,flags.logdir)
    if not os.path.isdir(os.path.join(os.getcwd(), check_path)):
        os.makedirs(os.path.join(os.getcwd(), check_path))

    writer = SummaryWriter(log_dir=check_path)

    # Setup seeds
    torch.manual_seed(flags.seed)
    torch.cuda.manual_seed(flags.seed)
    np.random.seed(flags.seed)
    random.seed(flags.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    trfinput = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trftarget = transforms.Compose([transforms.ToTensor()])

    Reversetransform = transforms.ToPILImage(mode='RGB')

    phases = np.asarray(['train','val'])
    # task_sets = np.asarray(['seg','parts','edges','normals','saliency'])
    task_sets = np.asarray(['saliency'])
    # datasets = {}
    # dataloaders = {}

    # for phase in phases:
    #     datasets[phase] = {}
    #     dataloaders[phase] = {}

    #     if phase == 'train':
    #         augment_task = get_training_augmentation()
    #     else:
    #         augment_task = get_validation_augmentation()

    #     for task in task_sets:
    #         if task == 'normals':
    #             augmentation = None
    #             trftarget_task = None
    #         else :
    #             augmentation = augment_task
    #             trftarget_task = trftarget

    #         # datasets[phase][task] = VOCSegmentation(root=root_path, image_set=phase, task_set = task, augmentation=augmentation, transform=trfinput, target_transform=trftarget_task)
    #         datasets[phase][task] = VOCSegmentation_new_fix_normal(root=root_path, image_set=phase, task_set = task, augmentation=augmentation, transform=trfinput, target_transform=trftarget_task)
    #         print("Length_{}_Images_{}:{}".format(phase, task, datasets[phase][task].__len__()))
    #         dataloaders[phase][task] = DataLoader(datasets[phase][task], batch_size=flags.batch_size, shuffle=True, num_workers=2)
    # print(datasets)
    # print(dataloaders)

    datasets, dataloaders = obtain_dataloader_from_kevis(phases, task_sets, flags.batch_size)

    taskname2kevis={'seg': 'semseg','parts':'human_parts','edges':'edge','normals':'normals','saliency':'sal'}

    latent_z_task = {}               #input code for the corresponding task
    for task in task_sets:
        z = torch.zeros([1, 512], dtype=torch.float)
        if task == 'seg':
            z[:, :102] = 1
            z[:, 102:] = 0
        elif task == 'parts':
            z[:, :102] = 0
            z[:, 102:204] = 1
            z[:, 204:] = 0
        elif task == 'edges':
            z[:, :204] = 0
            z[:, 204:306] = 1
            z[:, 306:] = 0
        elif task == 'normals':
            z[:, :306] = 0
            z[:, 306:408] = 1
            z[:, 408:] = 0
        elif task== 'saliency':
            z[:, :408] = 0
            z[:, 408:] = 1
        latent_z_task[task] = z

    # print(latent_z_task)

    # numclass = 21         # no of output channels, multi-heads is controlled here

    numclass_all={'seg':21, 'parts': 7, 'saliency':1,'normals':3, 'edges':1}

    # model = ResNetUNet2_2_no_adain(numclass).to(device)

    model = ResNetUNet2_2_no_adain(numclass_all[task_sets[0]]).to(device)

    print('\nNumber of parameters (in millions): {0:.3f}'.format(count_parameters(model) / 1e6))
    # exit()
    # print(model)

    # layer0-layer4 - enocoder part hence learning rate is low as pretrained imagenet weights are used for encoder initialization
    lr_backbone=1e-6
    lr_new=1e-4

    optimizer_ft = optim.Adam([
        {'params': model.layer0.parameters(), 'lr': lr_backbone},
        {'params': model.layer1.parameters(), 'lr': lr_backbone},
        {'params': model.layer2.parameters(), 'lr': lr_backbone},
        {'params': model.layer3.parameters(), 'lr': lr_backbone},
        {'params': model.layer4.parameters(), 'lr': lr_backbone},

        {'params': model.layer0_1x1.parameters(), 'lr': lr_new},
        {'params': model.layer1_1x1.parameters(), 'lr': lr_new},
        {'params': model.layer2_1x1.parameters(), 'lr': lr_new},
        {'params': model.layer3_1x1.parameters(), 'lr': lr_new},
        {'params': model.layer4_1x1.parameters(), 'lr': lr_new},

        {'params': model.upsample.parameters(), 'lr': lr_new},

        {'params': model.conv_up3.parameters(), 'lr': lr_new},
        {'params': model.conv_up2.parameters(), 'lr': lr_new},
        {'params': model.conv_up1.parameters(), 'lr': lr_new},
        {'params': model.conv_up0.parameters(), 'lr': lr_new},

        # {'params': model.conv_original_size0.parameters(), 'lr': lr_new},
        # {'params': model.conv_original_size1.parameters(), 'lr': lr_new},
        {'params': model.conv_original_size2.parameters(), 'lr': lr_new},

        {'params': model.conv_last.parameters(), 'lr': lr_new},
        # {'params': model.fcs.parameters(), 'lr': lr_new}

        # decrease lr for encoder in order not to permute
        # pre-trained weights with large gradients on training start
    ])

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, eps=1e-12)


    ## sgd optimizer added by guolei
    # train_params=[
    #     {'params': model.layer0.parameters(), 'lr': lr_backbone},
    #     {'params': model.layer1.parameters(), 'lr': lr_backbone},
    #     {'params': model.layer2.parameters(), 'lr': lr_backbone},
    #     {'params': model.layer3.parameters(), 'lr': lr_backbone},
    #     {'params': model.layer4.parameters(), 'lr': lr_backbone},

    #     {'params': model.layer0_1x1.parameters(), 'lr': lr_new},
    #     {'params': model.layer1_1x1.parameters(), 'lr': lr_new},
    #     {'params': model.layer2_1x1.parameters(), 'lr': lr_new},
    #     {'params': model.layer3_1x1.parameters(), 'lr': lr_new},
    #     {'params': model.layer4_1x1.parameters(), 'lr': lr_new},

    #     {'params': model.upsample.parameters(), 'lr': lr_new},

    #     {'params': model.conv_up3.parameters(), 'lr': lr_new},
    #     {'params': model.conv_up2.parameters(), 'lr': lr_new},
    #     {'params': model.conv_up1.parameters(), 'lr': lr_new},
    #     {'params': model.conv_up0.parameters(), 'lr': lr_new},

    #     # {'params': model.conv_original_size0.parameters(), 'lr': lr_new},
    #     # {'params': model.conv_original_size1.parameters(), 'lr': lr_new},
    #     {'params': model.conv_original_size2.parameters(), 'lr': lr_new},

    #     {'params': model.conv_last.parameters(), 'lr': lr_new},
    #     {'params': model.fcs.parameters(), 'lr': lr_new}

    #     # decrease lr for encoder in order not to permute
    #     # pre-trained weights with large gradients on training start
    # ]
    # optimizer_sgd = optim.SGD(train_params, lr=lr_new, momentum=p.TRAIN.MOMENTUM, weight_decay=p['wd'])
    # poly_scheduler = Poly_scheduler(optimizer_sgd, flags.epochs, momentum=0.9)

    criterion1 = nn.NLLLoss(ignore_index=255)
    criterion2 = nn.L1Loss()
    criterion_normals = NormalsLoss(normalize=True, size_average=True, norm=1)
    # criterion_normals = NormalsLoss(normalize=False, size_average=True, norm=1)
    criterion_edges = BalancedCrossEntropyLoss(size_average=True, pos_weight=0.95)
    criterion_sal = BalancedCrossEntropyLoss(size_average=True)
    criterion3 = nn.CrossEntropyLoss(ignore_index=255)

    start_epoch = 0
    start_iter = 0
    best_loss = 1e10

    if flags.best == 0 :
        if os.path.exists(flags.ckdir):
            path = os.path.join(flags.ckdir,'checkpoint.pth')
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["model_state"])
            optimizer_ft.load_state_dict(checkpoint["optimizer_state"])
            exp_lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            start_iter = checkpoint["iter"] + 1
            best_loss = checkpoint['best_loss']
            print("epoch_loaded",start_epoch-1)
    else:
        if os.path.exists(flags.ckdir):
            path1 = "train_model_epoch_{}.pth".format(flags.best)
            path = os.path.join(flags.ckdir, path1)
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["model_state"])
            optimizer_ft.load_state_dict(checkpoint["optimizer_state"])
            exp_lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            start_iter = checkpoint["iter"] + 1
            best_loss = checkpoint['best_loss']

    for epoch in range(start_epoch,flags.epochs):
        print('Epoch {}/{}'.format(epoch, flags.epochs - 1))
        print('-' * 10)

        since = time.time()

        epoch_loss = {}
        val_loss = 0
        count_task = 0

        np.random.shuffle(task_sets)                  #epochwise random shuffling of the tasks
        print(task_sets)

        for phase in phases:
            print('phase', phase)
            epoch_loss[phase] = {}
            if phase == 'train':
                param_count = 0
                for param_group in optimizer_ft.param_groups:
                    if (param_count == 0 or param_count == 6):
                        print("LR", param_group['lr'])
                    param_count = param_count + 1
                model.train()
            else:
                model.eval()

            for task in task_sets:

                epoch_samples = 0
                total_loss = 0
                
                print('task', task)

                for sample in dataloaders[phase][task]:
                    # inputs = inputs.to(device)
                    # labels = labels.to(device)
                    inputs = sample['image'].to(device)
                    labels = sample[taskname2kevis[task]].to(device)

                    latent_z = latent_z_task[task].repeat(inputs.size()[0],1)
                    latent_z = latent_z.to(device)
                    optimizer_ft.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model.forward(inputs,latent_z)
                        height,weight=outputs.size(2),outputs.size(3)

                        # if task == 'parts':
                        #     outputs=outputs.unsqueeze(1)
                        #     outputs=F.adaptive_avg_pool3d(outputs,(7,height,weight))
                        #     outputs=outputs.squeeze(1)
                        # elif task == 'normals':
                        #     outputs=outputs.unsqueeze(1)
                        #     outputs=F.adaptive_avg_pool3d(outputs,(3,height,weight))
                        #     outputs=outputs.squeeze(1)
                        # elif task == 'edges' or task == 'saliency':
                        #     outputs=outputs.unsqueeze(1)
                        #     outputs=F.adaptive_avg_pool3d(outputs,(1,height,weight))
                        #     outputs=outputs.squeeze(1)

                        # if task == 'seg':
                        #     outputs = outputs[:,:21,:,:]
                        # elif task == 'parts':
                        #     outputs = outputs[:,21:28,:,:]
                        # elif task == 'normals':
                        #     outputs = outputs[:,28:31,:,:]
                        # elif task == 'edges':
                        #     outputs = outputs[:,31,:,:].unsqueeze(1)
                        # elif task == 'saliency':
                        #     outputs = outputs[:,32,:,:].unsqueeze(1)

                        # import pdb;pdb.set_trace()
                        # print(outputs.size())
                        # break
                        # import pdb;pdb.set_trace()
                        if task == 'normals':
                            # outputs = 2 * outputs - 1
                            loss = 10.0*criterion_normals(outputs, labels)
                        elif task == 'seg' or task == 'parts':                   #computing cross-entropy-like-l1 loss
                            # outputs1 = torch.log(outputtoclasses(outputs, task, device).permute(0, 3, 1, 2))
                            # labels1 = labeltoclasses(labels, task, device).permute(0, 3, 1, 2)[:, 0, :, :]
                            # loss = criterion1(outputs, labels1)
                            # print(outputs.shape,labels1.shape)
                            labels=labels.squeeze(1).long()
                            loss = criterion3(outputs, labels)
                            if task == 'parts': 
                                loss = 2.0*loss
                        elif task == 'edges':
                            loss = 50.0*criterion_edges(outputs, labels)
                        elif task == 'saliency':
                            loss = 5.0*criterion_sal(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer_ft.step()
                    writer.add_scalar("loss/{}_{}_loss".format(phase,task),loss,start_iter)

                    start_iter += 1

                    epoch_samples += inputs.size(0)
                    total_loss += loss

                epoch_loss[phase][task] = total_loss / epoch_samples
                
                if phase == 'val':
                    val_loss+=epoch_loss[phase][task]
                    count_task+=1
                    if count_task == len(task_sets):
                        mean_val_loss = val_loss / count_task
                        exp_lr_scheduler.step(mean_val_loss)
                        print("Done")

        time_elapsed = time.time() - since

        print("epoch: '{}', epoch_loss: '{}', mean_val_loss: '{}',time_elapsed: '{}'"
              .format(epoch , epoch_loss, mean_val_loss, time_elapsed))

        if (mean_val_loss < best_loss):
            best_loss = mean_val_loss
            print("saving best model:", epoch)
            state = {"epoch": epoch,
                     "iter": start_iter,
                     "model_state": model.state_dict(),
                     "optimizer_state": optimizer_ft.state_dict(),
                     "scheduler_state": exp_lr_scheduler.state_dict(),
                     "best_loss": best_loss
                     }
            if not os.path.exists(flags.ckdir):
                os.mkdir(flags.ckdir)
            save_path1 = os.path.join(flags.ckdir, 'checkpoint.pth')
            torch.save(state, save_path1)

        if ((epoch + 1) % 10 == 0):
            state = {"epoch": epoch,
                     "iter": start_iter,
                     "model_state": model.state_dict(),
                     "optimizer_state": optimizer_ft.state_dict(),
                     "scheduler_state": exp_lr_scheduler.state_dict(),
                     "best_loss": best_loss
                     }
            if not os.path.exists(flags.ckdir):
                os.mkdir(flags.ckdir)
            p1 = "train_model_epoch_{}.pth".format(epoch)
            save_path2 = os.path.join(flags.ckdir, p1)
            torch.save(state, save_path2)

    writer.close()

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=IMG_SIZE[1], min_width=IMG_SIZE[2], always_apply=True, border_mode=0),
        albu.RandomCrop(height=IMG_SIZE[1], width=IMG_SIZE[2], always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        )
    ]
    return albu.Compose(train_transform)




def get_validation_augmentation():
    test_transform = [
            albu.PadIfNeeded(IMG_SIZE[1],IMG_SIZE[2],always_apply=True,border_mode=0)
    ]
    return albu.Compose(test_transform)

def reverse_transform_train(inp):

    inp = inp.numpy().transpose((0,2,3, 1))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = inp.transpose((0,3,1,2))
    inp = torch.from_numpy(inp)

    return inp

def normal_loss(output, target):
    target1 = target.permute(0,2,3,1)
    one_label1 = torch.ones_like((target1))
    zero_label1 = torch.zeros_like((target1))
    target2 = torch.sqrt(torch.sum(torch.pow(target1,2),-1))
    target3 = target2[:, :, :, None]
    target4 = target3.repeat(1, 1, 1, 3)
    no_valid = torch.sum(torch.where(target4!=0,one_label1,zero_label1))
    target_valid = torch.where(target4!=0,target.permute(0,2,3,1),zero_label1)
    output_valid = torch.where(target4!=0,output.permute(0,2,3,1),zero_label1)
    loss = torch.sum(torch.abs(output_valid-target_valid))
    loss = loss/(no_valid + 1e-12)
    return loss

def normal_ize(bottom, dim=-1):
    qn = torch.norm(bottom, p=1, dim=dim).unsqueeze(dim=dim) + 1e-12

    return bottom.div(qn)

def outputtoclasses(image,task,device):
    image = image.permute(0,2,3,1)
    image = (image * 255).type(torch.cuda.FloatTensor)
    l1 = torch.zeros([image.size()[0],image.size()[1],image.size()[2],colourmap1[task].shape[0]],device=device)
    for i in range(0, colourmap1[task].shape[0]):
        b = torch.from_numpy(colourmap1[task][i,1:]).type(torch.cuda.FloatTensor)
        c = b[None,None,None,:]
        d = c.repeat(image.size()[0],image.size()[1],image.size()[2],1)
        l1[:,:,:,i] = torch.sum(torch.abs(image-d),dim=-1) + 1e-12
    l2 = torch.ones_like((l1))
    l3 = l2.div(l1)
    l4 = normal_ize(l3)
    l5 = torch.pow(l4,1)
    return l5

def labeltoclasses(image,task,device):
    image = image.permute(0,2,3,1)
    image = (image*255).type(torch.cuda.FloatTensor)
    a = torch.zeros([image.size()[0], image.size()[1], image.size()[2]], device=device)
    for i in range(0, colourmap1[task].shape[0]):
        b = torch.from_numpy(colourmap1[task][i, 1:]).type(torch.cuda.FloatTensor)
        c = torch.from_numpy(np.asarray(colourmap1[task][i,0]))
        a[(image == b).all(dim=-1)] = c
    a = a.unsqueeze(dim=-1).type(torch.cuda.LongTensor)
    return a

if __name__ == "__main__":
    flags = read_flags()
    main(flags)


