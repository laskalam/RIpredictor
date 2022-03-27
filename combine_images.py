import sys
import numpy as np
import pandas as pd
import h5py as h5
from netCDF4 import Dataset
from skimage.transform import resize
from configparser import ConfigParser

#Tempareture extrema
vmin, vmax = 175.00, 325.00
#-----
to_resize = True 
target_size = (224,224)

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def convert_image(image, vmin=vmin, vmax=vmax,
                  to_resize=to_resize,
                  target_size=target_size,
                  imagenet_mean=imagenet_mean,
                  imagenet_std=imagenet_std):
    """
    Rescale and resize the image. Depending on
    the model being used
    Input:
    ------
        image: NxM numpy array (map)
        Other parameters are defined in the config
        file.
    Use:
    ----
        convert_image(image)
    Output:
    -------
        image: LxPx3 numpy array. L and might be the same
               or different from N and M
    """

    image = np.clip(image, vmin, vmax)
    # scale to be [0,1)
    image = (image - vmin) / (vmax - vmin)
    # change image size
    if to_resize:
        image = resize(image, target_size)
    image = image[..., np.newaxis]
    ## convert to RGB channel
    image = np.clip(np.rint(np.repeat(image, 3, axis=-1)*255), 0, 255).astype(int)
    ## adjust to the initial weights
    return image

