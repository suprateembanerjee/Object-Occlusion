import time
import cv2
import torch
import numpy as np

def mask_yolo(det:torch.Tensor,
              s:str,
              img:torch.Tensor,
              names:list, 
              colors:list, 
              frame:np.ndarray,
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
    mask = np.zeros(frame.shape[:-1], dtype=np.uint8) #np.full(shape=image.shape, fill_value=255, dtype=np.uint8) #np.ones_like(frame)

    if len(det):
        # Rescale boxes from img_size to frame size
        det[:, :4] = funcs['scale_coords'](img.shape[2:], det[:, :4], frame.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        for x1, y1, x2, y2, conf, cls in det:

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            margin = 20
            
            if names[int(cls)] == 'person':
                # Reset Age of pixels
                mask[max(0, y1 - margin) : min(frame.shape[0], y2 + margin), max(0, x1 - margin) : min(frame.shape[1], x2 + margin)] = 1.

                # mask = cv2.normalize(mask.copy(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    return mask, s

