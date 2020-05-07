"""
Build datasets.

Data prediction type:
    complete: Edema and tumor (label 1, 2, 3);
    core: Necrotic, non-enhancing and enhancing tumor (label 1, 3);
    enhancing: Enhancing tumor (label 3);
"""
import os
import glob
import cv2
from transform import *

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def data_loader(args, mode):
    """data loader
    Args:
        args: Parameters;
        mode: The mode (train, valid or test);
    """
    # check data type
    if args.data == 'complete' or args.data == 'core' or args.data == 'enhancing':
        pass
    else:
        raise ValueError('Data type ERROR! Should be complete, core or enhancing.')

    # check mode
    if mode == 'train':
        shuffle = True
        dataset = TrainSet(args)
    elif mode == 'valid':
        shuffle = False
        dataset = ValidSet(args)
    elif mode == 'test':
        shuffle = False
        dataset = TestSet(args)
    else:
        raise ValueError('Mode ERROR! Should be train, valid or test.')

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=os.cpu_count(),
                            shuffle=shuffle,
                            drop_last=True)
    return dataloader


class TrainSet(Dataset):
    def __init__(self, args):
        self.data = args.data
        self.space = args.space // 2
        self.tumor_ratio = []

        self.img_root = args.img_root
        self.label_root = args.label_root
        # tumor ratio < 5%
        self.img_path1 = []
        self.label_path1 = []
        # tumor ratio > 5%
        self.img_path2 = []
        self.label_path2 = []

        # data augmentation
        self.img_transform = transforms.ColorJitter(brightness=0.2)
        self.transforms = Compose([
                    RandomVerticalFlip(),
                    RandomHorizontalFlip(),
                    RandomAffine(degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-0.2, 0.2)),
                    ElasticTransform(alpha=720, sigma=24)])
        self.totensor = transforms.ToTensor()

        # Data Handling Parameters
        if self.data == 'complete':
            self.patch_threshold = args.complete_threshold
            self.data_rate = args.complete_rate
        elif self.data == 'core':
            self.patch_threshold = args.core_threshold
            self.data_rate = args.core_rate
        else:
            self.patch_threshold = args.enhancing_threshold
            self.data_rate = args.core_rate

        # load images and labels
        img_path = sorted(glob.glob(os.path.join(self.img_root + '/*.jpg')))
        label_path = sorted(glob.glob(os.path.join(self.label_root + '/*.jpg')))

        for i in range(len(label_path)):
            label = cv2.imread(label_path[i], cv2.IMREAD_GRAYSCALE)

            if self.data == 'complete':
                tumor_area = (label > self.space).astype(np.uint8).sum()
            elif self.data == 'core':
                # Necrotic and non-enhancing tumor (label 1)
                l1 = (label > self.space).astype(np.uint8)
                l2 = (label < self.space * 3).astype(np.uint8)
                label1 = np.logical_and(l1, l2).astype(np.uint8)
                # Enhancing tumor (label 3)
                l1 = (label > self.space * 5).astype(np.uint8)
                l2 = (label < self.space * 7).astype(np.uint8)
                label2 = np.logical_and(l1, l2).astype(np.uint8)
                tumor_area = (np.logical_or(label1, label2).astype(np.uint8)).sum()
                del label1, label2
            else:
                # data prediction type = Enhancing
                # Enhancing tumor (label 3)
                l1 = (label > self.space * 5).astype(np.uint8)
                l2 = (label < self.space * 7).astype(np.uint8)
                tumor_area = (np.logical_and(l1, l2).astype(np.uint8)).sum()

            tumor_ratio = tumor_area / 18895  # brain area = 18895
            # self.tumor_ratio.append(tumor_ratio)  # decide threshold

            if tumor_ratio > self.patch_threshold:  # CHANGE FOR ENHANCING
                self.label_path2.append(label_path[i])
                self.img_path2.append(img_path[i])
            else:
                self.label_path1.append(label_path[i])
                self.img_path1.append(img_path[i])
        '''
        # decide threshold
        print('length of img_path1', len(self.img_path1))
        print('length of img_path2', len(self.img_path2))
        self.tumor_ratio = [self.tumor_ratio[i] for i in range(len(self.tumor_ratio)) if self.tumor_ratio[i] > 0]
        sorted(self.tumor_ratio)
        print('length of nonzero tumor ratio', len(self.tumor_ratio))
        # medium of tumor_ratio
        print('medium of tumor_ratio', sorted(self.tumor_ratio)[len(self.tumor_ratio) // 2])
        '''

    def __len__(self):
        return len(self.img_path1) + len(self.img_path2)

    def __getitem__(self, idx):
        # probability data_rate to be 1
        if np.random.choice(2, 1, p=[1-self.data_rate, self.data_rate]) == 0:
            idx = idx % len(self.img_path1)
            img_path = self.img_path1[idx]
            label_path = self.label_path1[idx]
        else:
            idx = idx % len(self.img_path2)
            img_path = self.img_path2[idx]
            label_path = self.label_path2[idx]

        # load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # load label
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if self.data == 'complete':
            label[label < self.space] = 0
            label[label >= self.space] = 1
            label = label.astype(np.uint8)
        elif self.data == 'core':
            # Necrotic and non-enhancing tumor (label 1)
            l1 = (label > self.space).astype(np.uint8)
            l2 = (label < self.space * 3).astype(np.uint8)
            label1 = np.logical_and(l1, l2).astype(np.uint8)
            # Enhancing tumor (label 3)
            l1 = (label > self.space * 5).astype(np.uint8)
            l2 = (label < self.space * 7).astype(np.uint8)
            label2 = np.logical_and(l1, l2).astype(np.uint8)
            label = np.logical_or(label1, label2).astype(np.uint8)
            del label1, label2
        else:
            # data prediction type = Enhancing
            # Enhancing tumor (label 3)
            l1 = (label > self.space * 5).astype(np.uint8)
            l2 = (label < self.space * 7).astype(np.uint8)
            label = np.logical_and(l1, l2).astype(np.uint8)

        img = Image.fromarray(img)
        label = Image.fromarray(label)
        img, label = self.transforms(img, label)
        img = self.img_transform(img)

        label = np.expand_dims(np.array(label), axis=0)
        label = label.astype(np.float32)
        label = np.concatenate((np.absolute(label-1), label), axis=0)
        label = torch.from_numpy(label)

        return self.totensor(img), label, img_path


class ValidSet(Dataset):
    def __init__(self, args):
        self.data = args.data
        self.space = args.space // 2
        self.img_root = args.img_root
        self.label_root = args.label_root
        self.totensor = transforms.ToTensor()

        # load images and labels
        self.img_path = sorted(glob.glob(os.path.join(self.img_root + '/*.jpg')))
        self.label_path = sorted(glob.glob(os.path.join(self.label_root + '/*.jpg')))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.label_path[idx], cv2.IMREAD_GRAYSCALE)

        if self.data == 'complete':
            label[label < self.space] = 0
            label[label >= self.space] = 1
            label = label.astype(np.uint8)
        elif self.data == 'core':
            # Necrotic and non-enhancing tumor (label 1)
            l1 = (label > self.space).astype(np.uint8)
            l2 = (label < self.space * 3).astype(np.uint8)
            label1 = np.logical_and(l1, l2).astype(np.uint8)
            # Enhancing tumor (label 3)
            l1 = (label > self.space * 5).astype(np.uint8)
            l2 = (label < self.space * 7).astype(np.uint8)
            label2 = np.logical_and(l1, l2).astype(np.uint8)
            label = np.logical_or(label1, label2).astype(np.uint8)
            del label1, label2
        else:
            # data prediction type = Enhancing
            # Enhancing tumor (label 3)
            l1 = (label > self.space * 5).astype(np.uint8)
            l2 = (label < self.space * 7).astype(np.uint8)
            label = np.logical_and(l1, l2).astype(np.uint8)

        img = Image.fromarray(img)
        label = Image.fromarray(label)

        label = np.expand_dims(np.array(label), axis=0)
        label = label.astype(np.float32)
        label = np.concatenate((np.absolute(label - 1), label), axis=0)
        label = torch.from_numpy(label)

        return self.totensor(img), label, self.img_path[idx]


class TestSet(Dataset):
    def __init__(self, args):
        self.data = args.data
        self.img_root = args.img_root
        self.totensor = transforms.ToTensor()

        self.img_path = sorted(glob.glob(os.path.join(self.img_root + '/*.jpg')))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx], cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(img)
        return self.totensor(img), self.img_path[idx]
