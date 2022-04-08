from __future__ import print_function
from __future__ import division
import torch
import torchvision
print("PyTorch Version: ",torch.__version__)
from torchsummary import summary
from utils import read_flags, mean_square_error, imsave
from data_loader import VOCSegmentation, VOCSegmentation_new, VOCSegmentation_new_fix_normal
from torchvision import transforms
from torch.utils.data import DataLoader
import PIL
import os
from PIL import Image
import numpy as np
import random
from torch import nn
import cv2
from model import ResNetUNet2_2_no_adain
import jaccard as evaluation
import torch.nn.functional as F
import imageio

## import kevis functions
from experiments.dense_predict.pascal_resnet import config as config
import fblib.dataloaders as dbs
from fblib.util.helpers import worker_seed
from fblib.util.custom_collate import collate_mil
from fblib.layers.loss import BalancedCrossEntropyLoss, SoftMaxwithLoss, NormalsLoss, DepthLoss
from fblib.util.model_resources.num_parameters import count_parameters


# IMG_SIZE = (3,256,256)
root_path = "/raid/guolei/fully-seg/PASCAL_MT"       #change this
phase = 'val'

colourmap1 = {}

# # colourmap['seg'] is the ground truth array for the semantic segmentation classes
# colourmap = np.loadtxt('colourmap_21index.txt',delimiter='\t')
# colourmap1['seg'] = np.zeros_like(colourmap)
# colourmap1['seg'][:,0] = colourmap[:,0]
# colourmap1['seg'][:,1] = colourmap[:,3]
# colourmap1['seg'][:,2] = colourmap[:,2]
# colourmap1['seg'][:,3] = colourmap[:,1]

# # colourmap['parts'] is the ground truth array for the human parts segmentation classes
# colourmap = np.loadtxt('colourmap7.txt',delimiter='\t')
# colourmap1['parts'] = np.zeros_like(colourmap)
# colourmap1['parts'][:,0] = colourmap[:,0]
# colourmap1['parts'][:,1] = colourmap[:,3]
# colourmap1['parts'][:,2] = colourmap[:,2]
# colourmap1['parts'][:,3] = colourmap[:,1]

INFER_FLAGVALS={'seg':cv2.INTER_NEAREST,
                 'parts':cv2.INTER_NEAREST, 
                 'edges':cv2.INTER_LINEAR, 
                 'normals':cv2.INTER_LINEAR, 
                 'saliency':cv2.INTER_LINEAR
                }

def obtain_dataloader_from_kevis(phases, task_sets, batch_size):
    # class EmptyParser():
    #     def parse_args():
    #         return
    # import argparse
    # argparse.ArgumentParser = EmptyParser
    

    p = config.create_config()
    # print(p)
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
                dataset, dataloader=get_test_loader(p, 'PASCALContext', transforms_ts, infer=True)
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


def main(flags):

    # Setup seeds
    torch.manual_seed(flags.seed)
    torch.cuda.manual_seed(flags.seed)
    np.random.seed(flags.seed)
    random.seed(flags.seed)

    print('phase',phase)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    trfinput = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trftarget = transforms.Compose([transforms.ToTensor()])

    Reversetransform = transforms.ToPILImage(mode='RGB')

    Reversetransform = transforms.ToPILImage(mode='RGB')

    # task_sets = np.asarray(['seg', 'parts', 'normals', 'edges', 'saliency'])
    task_sets = np.asarray([flags.task])
    datasets = {}
    dataloaders = {}


    # datasets[phase] = {}
    # dataloaders[phase] = {}

    # for task in task_sets:
    #     if task == 'normals':
    #         trftarget_task = None
    #     else:
    #         trftarget_task = trftarget

    #     datasets[phase][task] = VOCSegmentation_new_fix_normal(root=root_path, image_set=phase, task_set=task,transform=trfinput,
    #                                                 target_transform=trftarget_task)
    #     print("Length_{}_Images_{}:{}".format(phase, task, datasets[phase][task].__len__()))
    #     dataloaders[phase][task] = DataLoader(datasets[phase][task], batch_size=flags.batch_size, shuffle=False,
    #                                               num_workers=2)
    # print(datasets)
    # print(dataloaders)

    datasets, dataloaders = obtain_dataloader_from_kevis([phase], task_sets, flags.batch_size)

    taskname2kevis={'seg': 'semseg','parts':'human_parts','edges':'edge','normals':'normals','saliency':'sal'}


    latent_z_task = {}                 #input code for the corresponding task
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
        elif task == 'saliency':
            z[:, :408] = 0
            z[:, 408:] = 1
        latent_z_task[task] = z

    # print(latent_z_task)

    # numclass = 3                # no of output channels
    # numclass = 21                # no of output channels

    numclass_all={'seg':21, 'parts': 7, 'saliency':1,'normals':3, 'edges':1}

    # model = ResNetUNet(numclass).to(device)
    # model = ResNetUNet2_2_no_adain(numclass).to(device)

    model = ResNetUNet2_2_no_adain(numclass_all[task_sets[0]]).to(device)

    print('\nNumber of parameters (in millions): {0:.3f}'.format(count_parameters(model) / 1e6))

    if flags.best == 0 :
        if os.path.exists(flags.ckdir):
            path = os.path.join(flags.ckdir,'checkpoint.pth')
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["model_state"])
            epoch = checkpoint["epoch"]
            print("epoch",epoch)
    else:
        if os.path.exists(flags.ckdir):
            # path1 = "train_model_epoch_{}.pth".format(flags.best)
            path1 = "single_task_{}.pth".format(task_sets[0])
            path = os.path.join(flags.ckdir, path1)
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["model_state"])

    model.eval()  # Set model to evaluate mode
    model.to(device)

    with torch.no_grad():

        for task in task_sets:
            print()
            print()
            print("##################################################################")
            print()
            print()
            
            epoch_samples = 0

            if task == 'seg' or task == 'parts':
                if task == 'seg':
                    n_classes = 21
                else:
                    n_classes = 7

                tp = [0] * n_classes
                fp = [0] * n_classes
                fn = [0] * n_classes
            if task == 'normals':
                deg_diff = []
            if task == 'saliency':

                length_jac = len(dataloaders[phase][task])
                mask_thres = np.linspace(0.2, 0.9, 15)

                eval_result= dict()
                eval_result['all_jaccards'] = np.zeros((length_jac, len(mask_thres)))
                eval_result['prec'] = np.zeros((length_jac, len(mask_thres)))
                eval_result['rec'] = np.zeros((length_jac, len(mask_thres)))

            if task == 'edges':

                length_jac = len(dataloaders[phase][task])
                mask_thres = np.linspace(0, 1, 100)

                eval_result = dict()
                eval_result['prec'] = np.zeros((length_jac, len(mask_thres)))
                eval_result['rec'] = np.zeros((length_jac, len(mask_thres)))

            for i, samples in enumerate(dataloaders[phase][task]):
                # inputs = samples[0].to(device)
                # labels = samples[1].to(device)

                inputs = samples['image'].to(device)
                # print(inputs.max(), inputs.min(), inputs.mean(), inputs.dtype)
                labels = samples[taskname2kevis[task]].to(device)

                latent_z = latent_z_task[task].repeat(inputs.size()[0], 1)
                latent_z = latent_z.to(device)
                outputs = model(inputs,latent_z)

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

                outputs_ori=outputs
                # import pdb;pdb.set_trace()
                if task == 'seg' or task =='parts':
                    # gt, labels1 = rgbtolabel(labels.to(torch.device('cpu')),task)
                    # mask, outputs1 = rgbregressedtolabel(outputs.to(torch.device('cpu')),task)
                    gt=labels.squeeze(1).cpu().numpy()
                    mask = outputs.detach().max(dim=1)[1].cpu().numpy()
                    outputs1 = outputs.detach().cpu().numpy()

                    assert (gt.shape==mask.shape)

                    valid = (gt != 255)

                    for i_part in range(0, n_classes):
                        tmp_gt = (gt == i_part)
                        tmp_pred = (mask == i_part)
                        tp[i_part] += np.sum(tmp_gt & tmp_pred & valid)
                        fp[i_part] += np.sum(~tmp_gt & tmp_pred & valid)
                        fn[i_part] += np.sum(tmp_gt & ~tmp_pred & valid)

                # if task == 'normals':
                #     # outputs = 2 * outputs - 1

                #     labels1 = to_numpy(labels.to(torch.device('cpu')))
                #     outputs1 = to_numpy(outputs.to(torch.device('cpu')))

                #     outputs1 = normal_ize(outputs1)

                #     labels1[labels1==255]=0.

                #     valid_mask = (np.linalg.norm(labels1, ord=2, axis=-1) != 0)
                #     outputs1[np.invert(valid_mask), :] = 0.
                #     labels1[np.invert(valid_mask), :] = 0.
                #     labels1 = normal_ize(labels1)

                #     deg_diff_tmp = np.rad2deg(np.arccos(np.clip(np.sum(outputs1 * labels1, axis=-1), a_min=-1, a_max=1)))
                #     deg_diff.extend(deg_diff_tmp[valid_mask])

                # if task == 'saliency' or task == 'edges':
                #     for j, thres in enumerate(mask_thres):
                #         labels_numpy = labels.to(torch.device('cpu')).numpy().transpose((0, 2, 3, 1))
                #         outputs_numpy = outputs.to(torch.device('cpu')).numpy().transpose((0, 2, 3, 1))
                #         gt = (labels_numpy > thres).astype(np.float32)
                #         mask_eval = (outputs_numpy > thres).astype(np.float32)
                #         if task == 'saliency':
                #             eval_result['all_jaccards'][i, j] = evaluation.jaccard(gt, mask_eval)
                #         eval_result['prec'][i, j], eval_result['rec'][i, j] = evaluation.precision_recall(gt, mask_eval)

                # for i in range(inputs.size(0)):
                #     input1 = reverse_transform(inputs[i].to(torch.device('cpu')))
                #     output1 = reverse_transform_outputs(outputs[i].to(torch.device('cpu')),task)

                #     if task == 'seg' or task == 'parts':
                #         output1 = np.hstack((output1, outputs1[i]))
                #         label1 = labels1[i]
                #     else:
                #         label1 = reverse_transform_outputs(labels[i].to(torch.device('cpu')),task)
                #     result = np.hstack((input1, output1, label1))
                #     add = '/RE%d.png' % (epoch_samples + 1 + i)
                #     folder = './Test1/' + task + 'Result'
                #     imsave(result, folder, folder + add)

                # epoch_samples += inputs.size(0)

                ## resize results as original images and save
                # import pdb;pdb.set_trace()
                # folder = './Test1/' + 'Result/' + flags.ckdir+ '/' + task+ '/'
                folder = flags.ckdir+ '/' + task+ '/'
                
                # break
                outputs=get_output(outputs_ori, task)
                # if task == 'edges':
                #     outputs=outputs[:,:,:,0]
                meta=samples['meta']
                if not os.path.exists(folder):
                        os.makedirs(folder)
                for jj in range(inputs.size(0)):

                    ## skip images with no meaningful labels
                    if len(labels[jj].unique())==1 and torch.unique(labels[jj])[0]==255:
                        continue

                    fname = meta['image'][jj]
                    # result = cv2.resize(outputs[jj], dsize=(meta['im_size'][1][jj], meta['im_size'][0][jj]),
                    #                         interpolation=INFER_FLAGVALS[task])
                    result = cv2.resize(outputs[jj], dsize=(int(meta['im_size'][1][jj]), int(meta['im_size'][0][jj])),
                                            interpolation=INFER_FLAGVALS[task])

                    imageio.imwrite(os.path.join(folder, fname + '.png'), result.astype(np.uint8))


            # if task == 'seg' or task == 'parts':

            #     jac = [0] * n_classes

            #     for i_part in range(0, n_classes):
            #         jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

            #     # Write results
            #     eval_result = dict()
            #     eval_result['jaccards_all_categs'] = jac
            #     eval_result['mIoU'] = np.mean(jac)

            # if task == 'normals':
            #     deg_diff = np.array(deg_diff)
            #     eval_result = dict()
            #     eval_result['mean'] = np.mean(deg_diff)
            #     eval_result['median'] = np.median(deg_diff)
            #     eval_result['rmse'] = np.mean(deg_diff ** 2) ** 0.5
            #     eval_result['11.25'] = np.mean(deg_diff < 11.25) * 100
            #     eval_result['22.5'] = np.mean(deg_diff < 22.5) * 100
            #     eval_result['30'] = np.mean(deg_diff < 30) * 100

            # if task == 'saliency' or task == 'edges':
            #     eval_result_new = dict()
            #     if task == 'saliency':
            #         eval_result_new['mIoUs'] = np.mean(eval_result['all_jaccards'], 0)
            #         eval_result_new['mIoU'] = np.max(eval_result_new['mIoUs'])

            #     eval_result_new['mPrec'] = np.mean(eval_result['prec'], 0)
            #     eval_result_new['mRec'] = np.mean(eval_result['rec'], 0)
            #     eval_result_new['F'] = 2 * eval_result_new['mPrec'] * eval_result_new['mRec'] / \
            #                        (eval_result_new['mPrec'] + eval_result_new['mRec'] + 1e-12)
            #     eval_result_new['maxF'] = np.max(eval_result_new['F'])
            #     eval_result=eval_result_new
            
            # print(eval_result)

            evaluate_all(folder, [task])  

def reverse_transform(inp):

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


def evaluate_all( folder, task_sets, overfit=False):
    print()
    print("*****************************************************")
    print()

    from fblib.dataloaders import pascal_context as pascal_context
    import json
    all_results=[]
    # import pdb;pdb.set_trace()
    for task in task_sets:
        # dataset_task=datasets[task]
        if task=='seg':
            from fblib.evaluation.eval_semseg import eval_semseg
            n_classes = 20
            # cat_names = VOC_CATEGORY_NAMES
            has_bg = True
            gt_set = 'val'
            db = pascal_context.PASCALContext(split=gt_set, do_edge=False, do_human_parts=False, do_semseg=True,
                                          do_normals=False, overfit=overfit)
            eval_results = eval_semseg(db, folder, n_classes=n_classes, has_bg=has_bg)
            print(eval_results)
        elif task=='parts':
            from fblib.evaluation.eval_human_parts import eval_human_parts
            gt_set = 'val'
            db = pascal_context.PASCALContext(split=gt_set, do_edge=False, do_human_parts=True, do_semseg=False,
                                          do_normals=False, do_sal=False, overfit=overfit)
            # eval_results = eval_human_parts(dataset_task, folder, task, n_parts=6)
            eval_results = eval_human_parts(db, folder)
            print(eval_results)
        elif task=='edges':
            eval_results = []
            print("evaluation of edges using matlab code base")
        elif task=='normals':
            from fblib.evaluation.eval_normals import eval_normals
            gt_set = 'val'
            db = pascal_context.PASCALContext(split=gt_set, do_edge=False, do_human_parts=False, do_semseg=False,
                                          do_normals=True, overfit=overfit)
            eval_results = eval_normals(db, folder)
            print(eval_results)
        elif task=='saliency':
            from fblib.evaluation.eval_sal import eval_sal
            split = 'val'
            db = pascal_context.PASCALContext(split=split, do_edge=False, do_human_parts=False, do_semseg=False,
                                          do_normals=False, do_sal=True, overfit=overfit)
            eval_results = eval_sal(db, folder, mask_thres=np.linspace(0.2, 0.9, 15))
            print('Results for Saliency Estimation')
            print('mIoU: {0:.3f}'.format(eval_results['mIoU']))
            print('maxF: {0:.3f}'.format(eval_results['maxF']))
        all_results.append(eval_results)
        
    fname= os.path.dirname(os.path.dirname(folder)) + '/' + task_sets[0] 

    with open(fname + '.json', 'w') as f:
        json.dump(all_results, f)


def get_output(output, task):
    output = output.permute(0, 2, 3, 1)
    if task == 'normals':
        output = (normal_ize_tensor(output, dim=3) + 1.0) * 255 / 2.0
    elif task in {'seg', 'parts'}:
        _, output = torch.max(output, dim=3)
    elif task in {'edges', 'saliency'}:
        output = (255 * 1 / (1 + torch.exp(-output)))
        output = output.squeeze(3)
    elif task in {'depth'}:
        pass
    else:
        raise ValueError('Select one of the valid tasks')

    return output.cpu().data.numpy()


def rgbtolabel(image,task):
    image = image.numpy().transpose((0,2,3,1))
    image = (image*255).astype(np.uint8)
    a = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    for i in range(0, colourmap1[task].shape[0]):
        a[(image == colourmap1[task][i, 1:]).all(axis=-1)] = colourmap1[task][i, 0]
    a = a.astype(int)
    colour_codes = colourmap1[task][:,1:]
    c = colour_codes[a].astype('float32')
    return a, c

def rgbregressedtolabel(image,task):
    image = image.numpy().transpose((0, 2, 3, 1))
    image = (image * 255).astype(np.uint8)
    l1 = np.zeros((image.shape[0],image.shape[1],image.shape[2],colourmap1[task].shape[0]))
    for i in range(0, colourmap1[task].shape[0]):
        b = colourmap1[task][i,1:]
        c = np.tile(b,(image.shape[0],image.shape[1],image.shape[2],1))
        l1[:,:,:,i] = np.sum(np.abs(image-c),axis=-1)
    l4 = np.argmin(l1,axis=-1)
    colour_labels = colourmap1[task][:, 0]
    colour_codes = colourmap1[task][:, 1:]
    cl = colour_labels[l4.astype(int)]
    cc = colour_codes[l4.astype(int)].astype('float32')
    return cl, cc

def reverse_transform_outputs(inp,task):

    inp = inp.detach().numpy().transpose((1, 2, 0))
    if task == 'normals':
        inp = ((inp + 1) * 255.0/2).astype(np.uint8)
    else:
        inp = (inp * 255).astype(np.uint8)
    return inp

def to_numpy(inp):

    inp1 = inp.detach().numpy().transpose((0,2,3,1))

    return inp1

def normal_ize(arr):
    arr_norm = np.linalg.norm(arr, ord=2, axis=-1)[..., np.newaxis] + 1e-12
    return arr / arr_norm

def normal_ize_tensor(bottom, dim=1):
    qn = torch.norm(bottom, p=2, dim=dim).unsqueeze(dim=dim) + 1e-12

    return bottom.div(qn)



if __name__ == "__main__":
    flags = read_flags()
    main(flags)


