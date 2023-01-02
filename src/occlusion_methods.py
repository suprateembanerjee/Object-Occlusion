import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import torch.nn as nn

def occlude_welford(history_array:numpy.ndarray, image:numpy.ndarray, background_img:numpy.ndarray, erode_dilate:bool=False) -> dict:
    '''
    Occludes objects in an image using historical distribution of each pixel updated online using Welford's method

    Parameters:
    history_array: Numpy array of shape [h, w, c, x] 
                   where x = [old mean, old std, old s, number of frames encountered before]
    image: Numpy array of shape [h, w, c] containing the present frame
    background_img: Numpy array of shape [h, w, c] containing the background image as computed till last frame
    erode_dilate: Boolean flag indicating whether to erode and dilate the mask (binary noise reduction technique)

    Return:
    Dictionary containing updated history, initial mask, updated background image, and final mask (same as initial if not eroded / dilated)
    '''

    if type(history_array) == type(None):
        history_array = np.full(shape=image.shape + (4,), fill_value=0.001, dtype=np.float32) # [mean, std, s, numpoints]

    t4 = time.time()

    print(history_array.shape)
    t5 = time.time()

    new_history = welford(history_array.copy(), image)

    num_std_diff = np.abs(new_history[:,:,:,0] - image) / new_history[:,:,:,1]
    t6 = time.time()
    mask = np.array(np.mean(num_std_diff, axis=2) < 3., dtype=np.float32)
    t7 = time.time()
    mask_out = mask.copy()

    if erode_dilate:
        ed_shape = cv2.MORPH_ELLIPSE

        erosion_size = 1
        element = cv2.getStructuringElement(ed_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                               (erosion_size, erosion_size))

        eroded = cv2.erode(mask, element)

        dilation_size = 2
        element = cv2.getStructuringElement(ed_shape, (2 * dilation_size + 1, 2 * dilation_size + 1),
                               (dilation_size, dilation_size))
        eroded_dilated = cv2.dilate(eroded, element)

        erosion_size = 1
        element = cv2.getStructuringElement(ed_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                               (erosion_size, erosion_size))

        mask_out = cv2.erode(eroded_dilated, element)


    background_img[np.nonzero(mask_out)] = image[np.nonzero(mask_out)]
    t8 = time.time()

    history_array = welford(history_array, background_img)

    mask = cv2.normalize(mask, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_out = cv2.normalize(mask_out, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    mask_out = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)

    return {'history': history_array, 'mask': mask, 'background image': background_img, 'final mask': mask_out}

def occlude_pixel_history(timeout:int, history_array:numpy.ndarray, image:numpy.ndarray, background_img:numpy.ndarray) -> dict:
    '''
    Occludes objects in an image using the historical distribution (max 200 (latest) encountered values) of each pixel

    Parameters:
    timeout: Time to wait before updating background, used to initially populate history
    history_array: Numpy array of shape [h, w, c, x] 
                   where x = queue of encountered values per channel maxing out at 200
    image: Numpy array of shape [h, w, c] containing the present frame
    background_img: Numpy array of shape [h, w, c] containing the background image as computed till last frame

    Return:
    Dictionary containing updated history array, mask, and updated background image
    '''

    if type(history_array) == type(None):
        history_array = np.full(shape=image.shape + (1,), fill_value=0.001, dtype=np.float32)

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