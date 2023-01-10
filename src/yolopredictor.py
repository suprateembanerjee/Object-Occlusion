# Project: Object Occlusion (2023)
# Author: Suprateem Banerjee (Github: @suprateembanerjee)

import cv2
import torch
import numpy as np
import torch.nn as nn
import time
import sys

def update_compatibility(model) -> None:
    '''
    Updates compatibility of YOLO model for different torch versions.
    Identical to similar mechanism used in official YOLOv7 repository.
    '''

    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

class YoloPredictor:
	'''
	This class was built to enable YOLO predictions on an image in a simple manner.

	Usage:
	1. Download YOLOv7 code from official github repo, store it in some {PATH_TO_YOLO}
	2. Download YOLOv7 weights (there are several versions) from official github repo, name it {WEIGHTS}
	3. Initialize predictor [predictor = YoloPredictor({PATH_TO_YOLO},{WEIGHTS})]
	4. Predict [predictor.predict(image)]
	'''

	def __init__(self,
				path_to_yolo:str,
				weights:str = 'yolov7.pt', 
				view_img:bool = True,
				save_txt:bool = True,
				imgsz:int = 640,
				trace:bool = False,
				project:str = 'runs/detect',
				name:str = 'exp',
				exist_ok:bool = False,
				device:str = 'cpu',
				augment:bool = False,
				conf_thres:float = 0.25,
				iou_thres:float = 0.45,
				classes:list = None,
				agnostic_nms:bool = False,
				save_conf:bool = False,
				save_img:bool = True,
				classify:bool = False,
				save_separate:bool = True):

		sys.path.insert(0, path_to_yolo)

		from utils.datasets import letterbox
		from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, set_logging
		from utils.plots import plot_one_box
		from utils.torch_utils import select_device, load_classifier, TracedModel

		self.funcs = {}
		self.funcs['letterbox'] = letterbox
		self.funcs['non_max_suppression'] = non_max_suppression
		self.funcs['apply_classifier'] = apply_classifier
		self.funcs['scale_coords'] = scale_coords
		self.funcs['plot_one_box'] = plot_one_box

		self.weights = f'{path_to_yolo}/{weights}'
		self.view_img = view_img
		self.save_txt = save_txt
		self.imgsz = imgsz
		self.trace = trace
		self.project = project
		self.name = name
		self.exist_ok = exist_ok
		self.device = device
		self.augment = augment
		self.conf_thres = conf_thres
		self.iou_thres = iou_thres
		self.classes = classes
		self.agnostic_nms = agnostic_nms
		self.save_conf = save_conf
		self.save_img = save_img
		self.classify = classify
		self.save_separate = save_separate

		# self.webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

		# Initialize
		set_logging()
		self.device = select_device(self.device)
		self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

		# Load model
		self.model = torch.load(self.weights, map_location=self.device)['model'].float().fuse().eval()
		update_compatibility(self.model)
		self.stride = int(self.model.stride.max())  # model stride
		self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

		if self.trace:
			self.model = TracedModel(self.model, self.device, self.img_size)

		if self.half:
			self.model.half()  # to FP16

		# Second-stage classifier
		if self.classify:
			self.modelc = load_classifier(name='resnet101', n=2)  # initialize
			self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

		# Set Dataloader
		self.vid_path = None
		self.vid_writer = None

		# Get names and colors
		self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
		self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

		# Run inference
		if self.device.type != 'cpu':
			self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
		self.old_img_w = old_img_h = imgsz
		self.old_img_b = 1

		self.background_img = None
		self.age_array = None

	def predict(self, frame:np.ndarray, img_size:int=640, stride:int=32) -> tuple:
		'''
	    Uses YOLO to detect objects in an image.
	    
	    Parameters:
	    frame: A video frame of shape (h, w, c)
	    img_size: Size of the image on which to perform detection
	    stride: Stride to be used by YOLO for detections
	    
	    Return:
	    Tuple containing detections, reshaped image, and timestamps
	    '''


		img = self.funcs['letterbox'](frame.copy(), img_size, stride=stride)[0]

		# Convert
		img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
		img = np.ascontiguousarray(img)

		img = torch.from_numpy(img).to(self.device)
		img = self.img.half() if self.half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		if self.background_img is None:
			self.background_img = frame.copy()
			self.age_array = np.zeros(frame.shape[:2])

		self.age_array += 1.

		t1 = t2 = t3 = t4 = t5 = t6 = t7 = t8 = time.time()

		# Inference
		t1 = time.time()
		with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
			pred = self.model(img, augment=self.augment)[0]
		t2 = time.time()

		# # Apply NMS
		pred = self.funcs['non_max_suppression'](pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
		t3 = time.time()

		# Apply Classifier
		if self.classify:
			pred = self.funcs['apply_classifier'](pred, modelc, img, frame)

		# Process detections
		det = pred[0]

		timestamps = {'t1':t1, 't2':t2, 't3':t3}

		return det, img, timestamps