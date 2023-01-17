# Project: Object Occlusion (2023)
# Author: Suprateem Banerjee (Github: @suprateembanerjee)

import time
import cv2
import torch
import numpy as np

from occlude_utils import welford, welford_next

def occlude_welford(history_array:np.ndarray, image:np.ndarray, background_img:np.ndarray, erode_dilate:bool=False) -> dict:
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

    t4 = t5 = t6 = t7 = t8 = time.time()

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

    # mask = cv2.normalize(mask, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # mask_out = cv2.normalize(mask_out, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    # mask_out = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)

    timestamps = {'t4':t4, 't5':t5, 't6':t6, 't7':t7, 't8':t8}

    return {'history': history_array, 'mask': mask, 'background image': background_img, 'final mask': mask_out, 'timestamps': timestamps}

def occlude_pixel_history(timeout:int, history_array:np.ndarray, image:np.ndarray, background_img:np.ndarray) -> dict:
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

    t4 = t5 = t6 = t7 = t8 = time.time()

    image_reshaped = image.reshape(image.shape + (1,))
    valid_bg = image_reshaped.copy()
    mask_out = np.ones_like(image)

    if timeout < 0:
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
    timestamps = {'t4':t4, 't5':t5, 't6':t6, 't7':t7, 't8':t8}

    return {'history': history_array, 'mask': mask_out, 'background image': background_img, 'timestamps': timestamps}

def occlude_yolo(det:torch.Tensor, 
                  s:str, 
                  img:torch.Tensor, 
                  names:list, 
                  colors:list, 
                  frame:np.ndarray, 
                  mask:np.ndarray, 
                  age_array:np.ndarray, 
                  background_img:np.ndarray, 
                  funcs:dict) -> dict:
    '''
    Occludes objects in an image using detection via YOLOv7 and masking the detected bounding box using present background

    Parameters:
    det: Tensor contatining detections by YOLOv7 [torch.float32]
    s: String for terminal out
    img: Tensor containing rescaled and reshaped image for YOLOv7 detections [torch.float32]
    names: List of prediction class names
    colors: List of prediction class colors
    frame: present frame from video [np.uint8]
    mask: present mask for background update [np.float64]
    age_array: age array denoting number of frames before when an object has not been detected in a pixel [np.float64]
    background_img: Numpy array of shape [h, w, c] containing the background image as computed till last frame [np.uint8]
    funcs: YOLOv7 functions which are required for processing

    Return:
    Dictionary containing string output, mask, updated background image, masked background update, and age array
    '''


    image = frame.copy()

    if len(det):
        # Rescale boxes from img_size to frame size
        det[:, :4] = funcs['scale_coords'](img.shape[2:], det[:, :4], frame.shape).round()

        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        for x1, y1, x2, y2, conf, cls in det:

            label = f'{names[int(cls)]} {conf:.2f}'
            funcs['plot_one_box']((x1, y1, x2, y2), frame, label=label, color=colors[int(cls)], line_thickness=1)

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            margin = 10
            
            # Add bbox to image
            if names[int(cls)] == 'person':
                # Reset Age of pixels
                mask[max(0, y1 - margin) : min(frame.shape[0], y2 + margin), max(0, x1 - margin) : min(frame.shape[1], x2 + margin)] = 0.

                label = f'{names[int(cls)]} {conf:.2f}'
                funcs['plot_one_box']((x1, y1, x2, y2), frame, label=label, color=colors[int(cls)], line_thickness=1)
            
            else:
                # Associate objects with people
                if np.any(mask[y1:y2,x1:x2], where=[0.]):
                    mask[max(0, y1 - margin) : min(frame.shape[0], y2 + margin), max(0, x1 - margin) : min(frame.shape[1], x2 + margin)] = 0.

        age_array *= mask
        background_update = np.tile((age_array > 20).astype(np.uint8), (3, 1, 1)).transpose(1,2,0) * image
        mask = np.sum(background_update, axis=2)
        background_img[np.nonzero(mask)] = background_update[np.nonzero(mask)]

    mask_out = cv2.normalize(mask.copy(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    mask_out = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)

    return {'string': s, 'mask': mask_out, 'background image': background_img, 'background update': background_update, 'age array': age_array}

def occlude_histogram():
    pass