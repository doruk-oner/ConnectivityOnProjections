import math
import numpy as np
from random import random

def get_shape(images):
    if isinstance(images, (list,tuple)):
        image_shape = images[0].shape
    else:
        image_shape = images.shape
    return image_shape

def _crop(image_shape, shape=(512,512), method='random'):

    for image_dim, dim in zip(image_shape, shape):
        assert image_dim > dim, "image is smaller than the cropped shape!"

    if method=='random':
        f = lambda image_dim, crop_dim: np.random.randint(0, image_dim-crop_dim + 1)
    elif method=='center':
        f = lambda image_dim, crop_dim: (image_dim - crop_dim)//2
    elif method=='upperleft':
        f = lambda image_dim, crop_dim: 0
    else:
        raise ValueError("unknown crop method {}".format(method))

    inits = [f(image_shape[i], shape[i]) for i in range(len(shape))]
    tuple_slice = tuple(slice(init,init + dim) for init,dim in zip(inits, shape))
    return tuple_slice

def crop(images, shape=(512,512), method='random'):
    """ Crop image(s)

    If param images is an iterable (a.k.a list, tuple, ..)
    the method applys the exact same transformation to all
    elements in the iterable.

    Parameters
    ----------
    images : numpy.ndarray or iterable of numpy.ndarray (N,D1,D2,..,Channels)
        image or list of images.
    shape : tuple or list
        crop size
    method : str
        'random' or 'center' or 'upperleft'

    Return
    ------
    numpy.ndarray or iterable of numpy.ndarray
    """
    image_shape = get_shape(images)

    tuple_slice = _crop(image_shape, shape, method)
    if isinstance(images, (list,tuple)):
        return [image[tuple_slice] for image in images] + [tuple_slice]

    return images[tuple_slice], tuple_slice

