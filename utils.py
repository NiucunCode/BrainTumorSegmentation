import os
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as util

from uNet import *


def dice_coefficient_loss(pred, target, backprop=True):
    """Calculate dice coefficient loss.
    Args:
        pred: prediction label;
        target: target label;
        backprop: if False, then need to generalize;
    Return:
        dice_loss: dice coefficient loss;
    """
    smooth = 1.0
    class_num = 2
    dice_loss = 0

    if backprop:
        for i in range(class_num):
            iflat = pred[:, i, :, :]
            tflat = target[:, i, :, :]
            intersection = (iflat * tflat).sum()
            loss = 1.0 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
            dice_loss += loss
        dice_loss /= class_num
    else:
        target = np.array(target.detach().argmax(1))
        if len(pred.shape) > 3:
            pred = np.array(pred.detach()).argmax(1)
        for i in range(class_num):
            iflat = (pred == i).astype(np.uint8)
            tflat = (target == i).astype(np.uint8)
            intersection = (iflat * tflat).sum()
            loss = 1.0 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
            dice_loss += loss
        dice_loss /= class_num

    return dice_loss


def load_model(args, mode):
    """load models
    Args:
        args: parameters;
        mode: train or test;
    Return:
        net: saved net;
        optimizer: saved optimizer;
        best_score: saved best score:
        start_epoch: saved start epoch;
    """
    # initialize model
    if args.model == 'unet':
        net = uNet(args.in_channel, args.class_num, drop_rate=args.drop_rate)
        net.apply(weights_init_normal)
    else:
        raise ValueError('Model ERROR!')

    # initialize optimizer
    if mode == 'train':
        resume = args.resume
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif mode == 'test':
        resume = True
        optimizer = None
    else:
        raise ValueError('Mode ERROR! Should be train or test.')

    # Model Load
    if resume:
        checkpoint = Checkpoint(net, optimizer)
        checkpoint.load(os.path.join(args.ckpt_root, args.model+'.pth'))
        best_score = checkpoint.best_score
        start_epoch = checkpoint.epoch + 1
    else:
        best_score = 0
        start_epoch = 1

    if args.use_gpu and torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net)

    if mode == 'train':
        return net, optimizer, best_score, start_epoch
    elif mode == 'test':
        return net


class Checkpoint:
    def __init__(self, model, optimizer=None, epoch=1, best_score=0):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.best_score = best_score

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        self.epoch = checkpoint["epoch"]
        self.best_score = checkpoint["best_score"]
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

    def save(self, path):
        torch.save({"model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                    "best_score": self.best_score}, path)


def get_crf_img(inputs, preds):
    """get CRF image
    Args:
        inputs: input test image;
        preds: prediction image;
    Return:
         crf: CRF image;
    http://graphics.stanford.edu/projects/drf/
    Adapted form: https://github.com/lucasb-eyer/pydensecrf/blob/master/pydensecrf/utils.py
    """
    for i in range(preds.shape[0]):
        img = inputs[i]
        softmax_prob = preds[i]
        unary = util.unary_from_softmax(softmax_prob)
        unary = np.ascontiguousarray(unary)
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], 2)  # 2 classes, width first then height
        d.setUnaryEnergy(unary)  # add unary
        # set pairwise potential
        feats = util.create_pairwise_gaussian(sdims=(10, 10), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        feats = util.create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20), img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        # inference
        Q = d.inference(5)  # inference 5 times
        result = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1])).astype(np.float32)

        if i == 0:
            crf = np.expand_dims(result, axis=0)
        else:
            result = np.expand_dims(result, axis=0)
            crf = np.concatenate((crf, result), axis=0)

    return crf


def erode_dilate(preds, kernel_size=7):
    """image processing -- erode or dilate
    Args:
        preds: prediction image;
        kernel_size: kernel size, default=7;
    Return:
        preds: processed image;
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    preds = preds.astype(np.uint8)
    for i in range(preds.shape[0]):
        img = preds[i]
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        preds[i] = img

    return preds


def post_process(args, inputs, preds, input_path=None, crf_flag=True, erode_dilate_flag=True,
                 save=True, overlap=True):
    """post process
    Args:
        args: parameters;
        inputs: input images;
        preds: prediction images;
        input_path: the path of input image;
        crf_flag: True to get CRF image;
        erode_dilate_flag: True to do erosion and dilation;
        save: True to save the result image;
        overlap: True to overlap;
    """
    # tensor to numpy image
    inputs = (np.array(inputs.squeeze()).astype(np.float32)) * 255
    inputs = np.expand_dims(inputs, axis=3)
    inputs = np.concatenate((inputs, inputs, inputs), axis=3)
    preds = np.array(preds)

    # conditional random field
    if crf_flag:
        preds = get_crf_img(inputs, preds)
    else:
        preds = preds.argmax(1)

    # erosion and dilation
    if erode_dilate_flag:
        preds = erode_dilate(preds, kernel_size=7)
    # save or not
    if not save:
        return preds

    outputs = preds * 255
    for i in range(outputs.shape[0]):
        foldername = input_path[i].split('/')[-2]
        basename = os.path.basename(input_path[i])
        output_folder = os.path.join(args.output_root, foldername)
        try:
            os.mkdirs(output_folder, exist_ok=True)
        except:
            pass
        output_path = os.path.join(output_folder, basename)

        if overlap:
            img = outputs[i]
            img = np.expand_dims(img, axis=2)
            zeros = np.zeros(img.shape)
            img = np.concatenate((zeros, zeros, img), axis=2)
            img = np.array(img).astype(np.float32)
            img = inputs[i] + img
            if img.max() > 0:
                img = (img / img.max()) * 255
            else:
                img = (img / 1) * 255
            cv2.imwrite(output_path, img)
        else:
            img = outputs[i]
            cv2.imwrite(output_path, img)

    return

