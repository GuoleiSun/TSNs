# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy.random as random
import numpy as np
import torch
import cv2
import math
import fblib.util.helpers as helpers


class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25), semseg=False, flagvals=None):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales
        self.semseg = semseg
        self.flagvals = flagvals

    def __call__(self, sample):

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'meta' in elem:
                continue

            tmp = sample[elem]

            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            assert(center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)
            if self.flagvals is None:
                if ((tmp == 0) | (tmp == 1)).all():
                    flagval = cv2.INTER_NEAREST
                elif 'gt' in elem and self.semseg:
                    flagval = cv2.INTER_NEAREST
                else:
                    flagval = cv2.INTER_CUBIC
            else:
                flagval = self.flagvals[elem]

            if elem == 'normals':
                # Rotate Normals properly
                in_plane = np.arctan2(tmp[:, :, 0], tmp[:, :, 1])
                nrm_0 = np.sqrt(tmp[:, :, 0] ** 2 + tmp[:, :, 1] ** 2)
                rot_rad= rot * 2 * math.pi / 360
                tmp[:, :, 0] = np.sin(in_plane + rot_rad) * nrm_0
                tmp[:, :, 1] = np.cos(in_plane + rot_rad) * nrm_0

            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            sample[elem] = tmp
        # print("scale: ", sample['image'].max(), sample['image'].min())
        return sample

    def __str__(self):
        return 'ScaleNRotate:(rot='+str(self.rots)+',scale='+str(self.scales)+')'


class FixedResize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        resolutions (dict): the list of resolutions
    """
    def __init__(self, resolutions=None, flagvals=None):
        self.resolutions = resolutions
        self.flagvals = flagvals
        if self.flagvals is not None:
            assert(len(self.resolutions) == len(self.flagvals))

    def __call__(self, sample):

        # Fixed range of scales
        if self.resolutions is None:
            return sample

        # print(self.flagvals)
        # print("before resize: ", sample['image'].max(), sample['image'].min())

        elems = list(sample.keys())
        for elem in elems:
            if 'meta' in elem or 'bbox' in elem:
                continue

            if elem in self.resolutions:
                if self.resolutions[elem] is None:
                    continue
                if isinstance(sample[elem], list):
                    if sample[elem][0].ndim == 3:
                        output_size = np.append(self.resolutions[elem], [3, len(sample[elem])])
                    else:
                        output_size = np.append(self.resolutions[elem], len(sample[elem]))
                    tmp = sample[elem]
                    sample[elem] = np.zeros(output_size, dtype=np.float32)
                    for ii, crop in enumerate(tmp):
                        if self.flagvals is None:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem])
                        else:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem], flagval=self.flagvals[elem])
                else:
                    if self.flagvals is None:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem])
                    else:
                        
                        # print("dddd", elem, self.flagvals[elem], sample[elem].shape, sample[elem].max(), sample[elem].min())
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem], flagval=self.flagvals[elem])
                        # print("dddd", elem, self.flagvals[elem], sample[elem].shape, sample[elem].max(), sample[elem].min())

                    if elem == 'normals':
                        N1, N2, N3 = sample[elem][:, :, 0], sample[elem][:, :, 1], sample[elem][:, :, 2]
                        Nn = np.sqrt(N1 ** 2 + N2 ** 2 + N3 ** 2) + np.finfo(np.float32).eps
                        sample[elem][:, :, 0], sample[elem][:, :, 1], sample[elem][:, :, 2] = N1/Nn, N2/Nn, N3/Nn
            else:
                del sample[elem]
        # print("resize: ", sample['image'].max(), sample['image'].min())
        # exit()
        return sample

    def __str__(self):
        return 'FixedResize:'+str(self.resolutions)


class RandomResize(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, scales=[0.5, 0.8, 1]):
        self.scales = scales

    def __call__(self, sample):

        # Fixed range of scales
        sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'meta' in elem or 'bbox' in elem:
                continue

            tmp = sample[elem]

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC

            tmp = cv2.resize(tmp, None, fx=sc, fy=sc, interpolation=flagval)

            sample[elem] = tmp

        return sample

    def __str__(self):
        return 'RandomResize:'+str(self.scales)


class FixedResizeRatio(object):
    """Fixed resize for the image and the ground truth to specified scale.
    Args:
        scales (float): the scale
    """
    def __init__(self, scale=None, flagvals=None):
        self.scale = scale
        self.flagvals = flagvals

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue

            if elem in self.flagvals:
                if self.flagvals[elem] is None:
                    continue

                tmp = sample[elem]
                tmp = cv2.resize(tmp, None, fx=self.scale, fy=self.scale, interpolation=self.flagvals[elem])

                sample[elem] = tmp

        return sample

    def __str__(self):
        return 'FixedResizeRatio: '+str(self.scale)


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                else:
                    tmp = sample[elem]
                    tmp = cv2.flip(tmp, flipCode=1)
                    sample[elem] = tmp

                if elem == 'normals':
                    sample[elem][:, :, 0] *= -1

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class NormalizeImage(object):
    """
    Return the given elements between 0 and 1
    """
    def __init__(self, norm_elem='image', clip=False):
        self.norm_elem = norm_elem
        self.clip = clip

    def __call__(self, sample):
        if isinstance(self.norm_elem, tuple):
            for elem in self.norm_elem:
                if np.max(sample[elem]) > 1:
                    sample[elem] /= 255.0
        else:
            if self.clip:
                sample[self.norm_elem] = np.clip(sample[self.norm_elem], 0, 255)
            if np.max(sample[self.norm_elem]) > 1:
                sample[self.norm_elem] /= 255.0
        return sample

    def __str__(self):
        return 'NormalizeImage'


class ToImage(object):
    """
    Return the given elements between 0 and 255
    """
    def __init__(self, norm_elem='image', custom_max=255.):
        self.norm_elem = norm_elem
        self.custom_max = custom_max

    def __call__(self, sample):
        if isinstance(self.norm_elem, tuple):
            for elem in self.norm_elem:
                tmp = sample[elem]
                sample[elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        else:
            tmp = sample[self.norm_elem]
            sample[self.norm_elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        return sample

    def __str__(self):
        return 'NormalizeImage'


class AddIgnoreRegions(object):
    """Add Ignore Regions"""

    def __call__(self, sample):

        for elem in sample.keys():
            tmp = sample[elem]

            if elem == 'normals':
                # Check areas with norm 0
                Nn = np.sqrt(tmp[:, :, 0] ** 2 + tmp[:, :, 1] ** 2 + tmp[:, :, 2] ** 2)

                tmp[Nn == 0, :] = 255.
                sample[elem] = tmp
            elif elem == 'human_parts':
                # Check for images without human part annotations
                if (tmp == 0).all():
                    tmp = 255 * np.ones(tmp.shape, dtype=tmp.dtype)
                    sample[elem] = tmp
            elif elem == 'depth':
                tmp[tmp == 0] = 255.
                sample[elem] = tmp

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            elif 'bbox' in elem:
                tmp = sample[elem]
                sample[elem] = torch.from_numpy(tmp)
                continue

            ## added by guolei, if elem is image, then skip. For image, we use totensor and normalize from pytorch
            if elem == 'image':
                continue

            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(tmp.astype(np.float32))

        return sample

    def __str__(self):
        return 'ToTensor'

class ToTensor_old(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            elif 'bbox' in elem:
                tmp = sample[elem]
                sample[elem] = torch.from_numpy(tmp)
                continue

            # ## added by guolei, if elem is image, then skip. For image, we use totensor and normalize from pytorch
            # if elem == 'image':
            #     continue

            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(tmp.astype(np.float32))

        return sample

    def __str__(self):
        return 'ToTensor'