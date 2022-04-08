import argparse
import os
import cv2
import numpy as np
IMG_SIZE = (3,256,256)

def read_flags():
    """Returns flags"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--epochs", default=300, type=int, help="Number of epochs")

    parser.add_argument("--batch_size", default=2, type=int, help="Batch size")

    parser.add_argument(
        "--logdir", default="logs", help="Tensorboard log directory")

    parser.add_argument(
        "--ckdir", default="checkpoints", help="Checkpoint directory")

    parser.add_argument(
        "--gpu", default="0", help="GPU_Number")

    parser.add_argument(
        "--seed", default=1337, help="Random Seed Generator")

    parser.add_argument(
        "--best", default=0, type=int, help="Model Selection")

    parser.add_argument(
        "--task", default='', help="task to be performed")

    flags = parser.parse_args()
    return flags

def imsave(image, path1, path2):
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(), path1)):
        os.makedirs(os.path.join(os.getcwd(), path1))

    cv2.imwrite(os.path.join(os.getcwd(), path2), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def mean_square_error(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    imageA = np.asarray(imageA)
    imageB = np.asarray(imageB)
    err = np.sum(np.square(imageA/255.0 - imageB/255.0))
    err /= float(IMG_SIZE[1] * IMG_SIZE[2])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


