"""
Used for data augmentation.
Need to do same transform on both image and label.

Adapted from torchvision/transforms/transforms.py
https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py

ElasticTransform cited form:
https://gist.github.com/oeway/2e3b989e0343f0884388ed7ed82eb3b0
__author__ = "Wei OUYANG"
__license__ = "GPL"
__version__ = "0.1.0"
__status__ = "Development"
"""
import random
import numbers
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import torchvision.transforms.functional as F
import collections
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            label (PIL Image or numpy.ndarray): Corresponding label to be converted to tensor.
        Returns:
            Tensor: Converted image.
            Tensor: Converted label.
        """
        return F.to_tensor(pic), F.to_tensor(label)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.
            label (PIL Image): Corresponding label to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
            PIL Image: Randomly flipped label.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.
            label (PIL Image): Corresponding label to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
            PIL Image: Randomly flipped label.
        """
        if random.random() < self.p:
            return F.vflip(img), F.vflip(label)
        return img, label

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands. This option is only available for ``pillow>=5.2.0``.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be rotated.
            label (PIL Image): Label to be rotated.
        Returns:
            PIL Image: Rotated image.
            PIL Image: Rotated label.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center, self.fill), \
               F.rotate(label, angle, self.resample, self.expand, self.center, self.fill)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image And int for grayscale) for the area
            outside the transform in the output image.(Pillow>=5.0.0)
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                    (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    # self.shear = [shear[0], shear[1], 0., 0.]
                    self.shear = shear
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = random.uniform(shears[0], shears[1])
                # shear = [random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [random.uniform(shears[0], shears[1]),
                         random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img, label):
        """
            img (PIL Image): Image to be transformed.
            label (PIL Image): Corresponding label to be transformed.

        Returns:
            PIL Image: Affine transformed image.
            PIL Image: Affine transformed label.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        return F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor), \
               F.affine(label, *ret, resample=self.resample, fillcolor=self.fillcolor)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)


def elastic_transform(image, alpha=1000, sigma=30, spline_order=1, mode='nearest', random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    # assert image.ndim == 3
    image = np.array(image)
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    # result = np.empty_like(image)
    result = map_coordinates(image, indices, order=spline_order, mode=mode).reshape(shape)
    '''
    for i in range(image.shape[2]):
        result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
    '''
    return Image.fromarray(result)


def random_num_generator(config, random_state=np.random):
    if config[0] == 'uniform':
        ret = random_state.uniform(config[1], config[2], 1)[0]
    elif config[0] == 'lognormal':
        ret = random_state.lognormal(config[1], config[2], 1)[0]
    else:
        print(config)
        raise Exception('unsupported format')
    return ret


class ElasticTransform(object):
    """Apply elastic transformation on a numpy.ndarray (H x W x C)
    """

    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image, label):
        if isinstance(self.alpha, collections.Sequence):
            alpha = random_num_generator(self.alpha)
        else:
            alpha = self.alpha
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(self.sigma)
        else:
            sigma = self.sigma
        return elastic_transform(image, alpha=alpha, sigma=sigma), elastic_transform(label, alpha=alpha, sigma=sigma)