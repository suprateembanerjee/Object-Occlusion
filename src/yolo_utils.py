import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import torch.nn as nn
import time

from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
	scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def update_compatibility(model) -> None:
    '''
    Updates compatibility of YOLO model for different torch versions
    '''

    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

class YoloPredictor:
	'''
	This class was built to use my occlusion mechanism with YOLO while maintaining code separation.
	This file can reside inside the main folder of YOLOv7, and use the detection, while the other occlusion methods stay separate
	and can make objects of YoloPredictor and use it to infer predictions
	'''

	def __init__(self,
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

		self.weights = weights
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

		# Directories
		# self.save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
		# (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

		# Initialize
		set_logging()
		self.device = select_device(self.device)
		self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

		# Load model
		self.model = torch.load(weights, map_location=device)['model'].float().fuse().eval()
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

		# if self.webcam:
		#     self.view_img = check_imshow()
		#     self.cudnn.benchmark = True  # set True to speed up constant image size inference
		#     self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride)
		# else:
		#     self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)

		# Get names and colors
		self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
		self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

		# Run inference
		if self.device.type != 'cpu':
			self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
		self.old_img_w = old_img_h = imgsz
		self.old_img_b = 1

		self.background_img = None
		self.age_array = None
		# self.history_array = None
		# self.timeout = 30

	def predict(self, frame, img_size=640, stride=32):

		img = letterbox(frame.copy(), img_size, stride=stride)[0]

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

		# p = Path(p)  # to Path
		# save_path = str(save_dir / p.name)  # img.jpg
		# txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
		
		mask = np.ones_like(self.age_array)
		image = frame.copy()

		# Inference
		t1 = time.time()
		with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
			pred = self.model(img, augment=self.augment)[0]
		t2 = time.time()

		# # Apply NMS
		pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
		t3 = time.time()

		# Apply Classifier
		if self.classify:
			pred = apply_classifier(pred, modelc, img, frame)

		# Process detections
		det = pred[0]

		return det, img, mask

	def occlude_yolo(self, det, s, img, frame, mask) -> dict:
	    '''
	    Occludes objects in an image using detection via YOLOv7 and masking the detected bounding box using present background

	    Parameters:
	    det: 
	    s:
	    names:
	    img:
	    frame:
	    age_array:
	    timeout: Time to wait before updating background, used to initially populate history
	    history_array: Numpy array of shape [h, w, c, x] 
	                   where x = queue of encountered values per channel maxing out at 200
	    image: Numpy array of shape [h, w, c] containing the present frame
	    background_img: Numpy array of shape [h, w, c] containing the background image as computed till last frame

	    Return:
	    Dictionary containing string output, mask, updated background image, and masked background update
	    '''

	    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
	    image = frame.copy()

	    if len(det):
	        # Rescale boxes from img_size to frame size
	        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

	        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh

	        # Print results
	        for c in det[:, -1].unique():
	            n = (det[:, -1] == c).sum()  # detections per class
	            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

	        # Write results
	        for x1, y1, x2, y2, conf, cls in det:

	            # if save_txt:  # Write to file
	            #     xywh = (xyxy2xywh(torch.tensor((x1, y1, x2, y2)).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
	            #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
	            #     with open(txt_path + '.txt', 'a') as f:
	            #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

	            if self.save_img or self.view_img:  # Add bbox to image
	                label = f'{self.names[int(cls)]} {conf:.2f}'
	                plot_one_box((x1, y1, x2, y2), frame, label=label, color=self.colors[int(cls)], line_thickness=1)

	            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
	            margin = 10
	            # Add bbox to image
	            if self.names[int(cls)] == 'person':
	                # Reset Age of pixels
	                mask[max(0, y1 - margin) : min(frame.shape[0], y2 + margin), max(0, x1 - margin) : min(frame.shape[1], x2 + margin)] = 0.

	                label = f'{self.names[int(cls)]} {conf:.2f}'
	                plot_one_box((x1, y1, x2, y2), frame, label=label, color=self.colors[int(cls)], line_thickness=1)
	            
	            else:
	                # Associate objects with people
	                if np.any(mask[y1:y2,x1:x2], where=[0.]):
	                    mask[max(0, y1 - margin) : min(frame.shape[0], y2 + margin), max(0, x1 - margin) : min(frame.shape[1], x2 + margin)] = 0.

	        self.age_array *= mask
	        background_update = np.tile((self.age_array > 20).astype(np.uint8), (3, 1, 1)).transpose(1,2,0) * image
	        mask = np.sum(background_update, axis=2)
	        self.background_img[np.nonzero(mask)] = background_update[np.nonzero(mask)]

	    mask_out = cv2.normalize(mask.copy(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
	    mask_out = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)

	    return {'string': s, 'mask': mask_out, 'background image': self.background_img, 'background update': background_update}
				