from PIL import Image
from torch.utils import data
import torchvision
from torchvision.datasets.utils import verify_str_arg
import os
import cv2
import numpy as np
import torch
import scipy.io as sio
from skimage.morphology import thin
import json

data_path = "ImageSets/Context_New"
data_path_new = "ImageSets/Context"    ## use the same split (only train/val) as kevis and Menelaos
input_path = "JPEGImages"
seg = "semseg/pascal-context"
parts = "human_superpartssymmetry"
edges = "edges_459blur"
normals = "normals_distill"
saliency = "sal_distill" 


# input_path = "JPEGImages"
# seg = "SemanticSegmentation/Context"
# parts = "human_superpartssymmetry"
# edges = "edges_459blur"
# normals = "NormalsDistill"
# saliency = "SaliencyDistill" 

# IMG_SIZE = (256, 256)
IMG_SIZE = (512, 512)

class VOCSegmentation(data.Dataset):

    def __init__(self,
                 root,
                 image_set='train',
                 task_set = 'seg',
                 augmentation=None,
                 transform=None,
                 target_transform=None):
        print("image size: ", IMG_SIZE)
        self.root = root
        self.image_set = verify_str_arg(image_set, "image_set",
                                        ("train", "val", "test"))
        self.task_set = verify_str_arg(task_set, "task_set",
                                        ("seg", "parts", "edges","normals","saliency"))
        self.augmentation = augmentation
        self.transform = transform
        self.target_transform = target_transform

        if task_set=='seg':
            label_path = seg
        elif task_set=='parts':
            label_path = parts
        elif task_set=='edges':
            label_path = edges
        elif task_set=='normals':
            label_path = normals
        elif task_set =='saliency':
            label_path = saliency
        
        image_dir = os.path.join(self.root, input_path)
        mask_dir = os.path.join(self.root, label_path)

        splits_dir = os.path.join(self.root, data_path)

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, mask) where mask is the ground truth for the corresponding task
        """
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE[1], IMG_SIZE[0]),interpolation=cv2.INTER_CUBIC)

        if self.task_set == 'seg' or self.task_set == 'parts' or self.task_set == 'edges' or self.task_set == 'saliency':
            mask = cv2.imread(self.masks[index])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST)

        if self.task_set == 'normals':
            mask = cv2.imread(self.masks[index])
            mask = 2 * cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - 1
            mask = cv2.resize(mask, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_CUBIC)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        if self.task_set == 'normals':
            mask = mask.transpose((2, 0, 1))
            mask = torch.from_numpy(mask.astype(np.float32))

        return image, mask


class VOCSegmentation_new(data.Dataset):
    ## use the same split (only train/val) as kevis and Menelaos

    def __init__(self,
                 root,
                 image_set='train',
                 task_set = 'seg',
                 augmentation=None,
                 transform=None,
                 target_transform=None):
        print("image size: ", IMG_SIZE)
        self.root = root
        self.image_set = verify_str_arg(image_set, "image_set",
                                        ("train", "val"))
        self.task_set = verify_str_arg(task_set, "task_set",
                                        ("seg", "parts", "edges","normals","saliency"))
        self.augmentation = augmentation
        self.transform = transform
        self.target_transform = target_transform

        if task_set=='seg':
            label_path = seg
        elif task_set=='parts':
            label_path = parts
        elif task_set=='edges':
            label_path = edges
        elif task_set=='normals':
            label_path = normals
        elif task_set =='saliency':
            label_path = saliency
        
        image_dir = os.path.join(self.root, input_path)
        mask_dir = os.path.join(self.root, label_path)

        splits_dir = os.path.join(self.root, data_path_new)

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, mask) where mask is the ground truth for the corresponding task
        """
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE[1], IMG_SIZE[0]),interpolation=cv2.INTER_CUBIC)

        if self.task_set == 'seg' or self.task_set == 'parts' or self.task_set == 'edges' or self.task_set == 'saliency':
            mask = cv2.imread(self.masks[index])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST)

        if self.task_set == 'normals':
            mask = cv2.imread(self.masks[index])
            mask = 2 * cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - 1
            mask = cv2.resize(mask, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_CUBIC)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        if self.task_set == 'normals':
            mask = mask.transpose((2, 0, 1))
            mask = torch.from_numpy(mask.astype(np.float32))

        return image, mask


class VOCSegmentation_new_fix_normal(data.Dataset):
    ## fix normals 

    def __init__(self,
                 root,
                 image_set='train',
                 task_set = 'seg',
                 augmentation=None,
                 transform=None,
                 target_transform=None):
        print("image size: ", IMG_SIZE)
        self.root = root
        self.image_set = verify_str_arg(image_set, "image_set",
                                        ("train", "val"))
        self.task_set = verify_str_arg(task_set, "task_set",
                                        ("seg", "parts", "edges","normals","saliency"))
        self.augmentation = augmentation
        self.transform = transform
        self.target_transform = target_transform

        if task_set=='seg':
            label_path = seg
        elif task_set=='parts':
            label_path = parts
        elif task_set=='edges':
            label_path = edges
        elif task_set=='normals':
            label_path = normals

            with open(os.path.join(self.root, 'db_info/nyu_classes.json')) as f:
                cls_nyu = json.load(f)
            with open(os.path.join(self.root, 'db_info/context_classes.json')) as f:
                cls_context = json.load(f)

            self.normals_valid_classes = []
            for cl_nyu in cls_nyu:
                if cl_nyu in cls_context and cl_nyu != 'unknown':
                    self.normals_valid_classes.append(cls_context[cl_nyu])

            # Custom additions due to incompatibilities
            self.normals_valid_classes.append(cls_context['tvmonitor'])
            # print('here: ',self.normals_valid_classes)

        elif task_set =='saliency':
            label_path = saliency
        
        image_dir = os.path.join(self.root, input_path)
        mask_dir = os.path.join(self.root, label_path)

        splits_dir = os.path.join(self.root, data_path_new)

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __len__(self):
        return len(self.images)

    def _load_normals_distilled(self, index):
        _tmp = np.array(Image.open(self.masks[index])).astype(np.float32)
        _tmp = 2.0 * _tmp / 255.0 - 1.0

        labels = sio.loadmat(os.path.join(self.root, 'pascal-context', 'trainval', self.images[index].split('/')[-1][:-4] + '.mat'))
        labels = labels['LabelMap']

        _normals = np.zeros(_tmp.shape, dtype=np.float)

        for x in np.unique(labels):
            if x in self.normals_valid_classes:
                _normals[labels == x, :] = _tmp[labels == x, :]

        return _normals

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, mask) where mask is the ground truth for the corresponding task
        """
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print("image size: ", image.shape)
        image = cv2.resize(image, (IMG_SIZE[1], IMG_SIZE[0]),interpolation=cv2.INTER_CUBIC)

        if self.task_set == 'seg' or self.task_set == 'parts' or self.task_set == 'edges' or self.task_set == 'saliency':
            mask = cv2.imread(self.masks[index])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST)

        if self.task_set == 'normals':
            # mask = cv2.imread(self.masks[index])
            # mask = 2 * cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - 1
            mask = self._load_normals_distilled(index)
            # print(mask.shape,mask.max(),mask.min(),IMG_SIZE)
            mask = cv2.resize(mask, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_CUBIC)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        if self.task_set == 'normals':
            mask = mask.transpose((2, 0, 1))
            mask = torch.from_numpy(mask.astype(np.float32))

        return image, mask

