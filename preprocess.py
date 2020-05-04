"""
Preprocess the dataset.

Dataset:
MRI images:
    shape: (H x W x N x C) = (240 x 240 x 155 x 4)
        channel_0: FLAIR;
        channel_1: T1;
        channel_2: T1c;
        channel_3: T2;

MRI labels:
    shape: (H x W x N) = (240 x 240 x 155)
    description:
        0: Background;
        1: Necrotic and non-enhancing tumor;
        2: Edema;
        3: Enhancing tumor;
"""
import os
import argparse

import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm


def normalize(img):
    """Normalize the input image.
    Args:
        img: input image;
    """
    return ((img / img.max()) * 255).astype(np.uint8)


def label_gray(label, space):
    """Distinguish labels.
    Args:
        label: input label;
        space: space between labels; (label * space)
            0: Background;
            1: Necrotic and non-enhancing tumor;
            2: Edema;
            3: Enhancing tumor;
    """
    if space * 3 > 255:
        space = 80
    return (label * space).astype(np.uint8)


def nii2jpg_img(input, output, channel):
    """Convert nii images to jpg.
    Args:
        input: input path;
        output: output path;
        channel: the channel of MRI data, see details above;
    """
    basename = os.path.basename(input).split('.')[0]
    try:
        os.makedirs(output, exist_ok=True)
    except:
        pass
    data = nib.load(input).get_fdata()[:, :, :, channel]
    img = normalize(data)

    # img.shape[2] = 155
    for i in tqdm(range(img.shape[2])):
        filename = os.path.join(output, basename + '_' + str(i) + '.jpg')
        gray_img = img[:, :, i]
        cv2.imwrite(filename, gray_img)
    # TODO: color


def nii2jpg_label(input, output, space):
    """Convert nii labels to jpg.
    Args:
        input: input path;
        output: output path;
        space: space between labels;
    """
    basename = os.path.basename(input).split('.')[0]
    try:
        os.makedirs(output, exist_ok=True)
    except:
        pass
    data = nib.load(input).get_fdata()
    label = label_gray(data, space)

    # label.shape[2] = 155
    for i in tqdm(range(label.shape[2])):
        filename = os.path.join(output, basename + '_' + str(i) + '.jpg')
        gray_label = label[:, :, i]
        cv2.imwrite(filename, gray_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=int, default=0, help="MRI images types.")
    parser.add_argument("--space", type=int, default=50, help="Space to distinguish labels.")
    parser.add_argument("--image_root", type=str, default='Task01_BrainTumor/imagesTr',
                        help="The root of dataset image.")
    parser.add_argument("--label_root", type=str, default='Task01_BrainTumor/labelsTr',
                        help="The root of dataset label.")
    args = parser.parse_args()

    label_output_root = 'train/label'
    if args.channel == 0:
        img_output_root = 'train/image_FLAIR'
    elif args.channel == 1:
        img_output_root = 'train/image_T1'
    elif args.channel == 2:
        img_output_root = 'train/image_T1c'
    elif args.channel == 3:
        img_output_root = 'train/image_T2'
    else:
        raise ValueError('Invalid channel!', args.channel)

    try:
        os.makedirs(img_output_root, exist_ok=True)
        os.makedirs(label_output_root, exist_ok=True)
    except:
        pass

    for path in tqdm(os.listdir(args.image_root)):
        nii2jpg_img(os.path.join(args.image_root, path), img_output_root, args.channel)

    for path in tqdm(os.listdir(args.label_root)):
        nii2jpg_label(os.path.join(args.label_root, path), label_output_root, args.space)
