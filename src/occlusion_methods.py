import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import torch.nn as nn

def occlude_welford(history_array, image, background_img) -> dict:

    if type(history_array) == type(None):
        history_array = np.full(shape=image.shape + (4,), fill_value=0.001, dtype=np.float64) # [mean, std, s, numpoints]

    t4 = time.time()

    print(history_array.shape)
    t5 = time.time()

    new_history = welford(history_array.copy(), image)

    num_std_diff = np.abs(new_history[:,:,:,0] - image) / new_history[:,:,:,1]
    t6 = time.time()
    mask = np.array(np.mean(num_std_diff, axis=2) < 3., dtype=np.float32)
    t7 = time.time()
    background_img[np.nonzero(mask)] = image[np.nonzero(mask)]
    t8 = time.time()

    history_array = welford(history_array, background_img)

    mask_out = cv2.normalize(mask.copy(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    mask_out = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)

    return {'history': history_array, 'mask': mask_out, 'background image': background_img}

def occlude_pixel_history(timeout, history_array, image, background_img) -> dict:

    if type(history_array) == type(None):
        history_array = np.full(shape=image.shape + (1,), fill_value=0.001, dtype=np.float64)

    t4 = time.time()

    image_reshaped = image.reshape(image.shape + (1,))
    valid_bg = image_reshaped.copy()
    mask_out = np.ones_like(image)

    if timeout < 0:
        print(f'270: history_array shape: {history_array.shape}')
        t5 = time.time()
        num_std_diff = np.abs(history_array.mean(axis=3) - image) / history_array.std(axis=3) # Expensive
        t6 = time.time()
        mask = np.array(np.mean(num_std_diff, axis=2) < 0.3, dtype=np.float32)
        t7 = time.time()
        background_img[np.nonzero(mask)] = image[np.nonzero(mask)]
        t8 = time.time()
        valid_bg = background_img.reshape(background_img.shape + (1,))
        mask_out = cv2.normalize(mask.copy(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        mask_out = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)

    history_array = np.append(history_array[:,:,:,int(history_array.shape[3] / 200):], valid_bg, axis=3)
    timeout -= 1

    return {'history': history_array, 'mask': mask_out, 'background image': background_img}