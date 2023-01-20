# Project: Object Occlusion (2023)
# Author: Suprateem Banerjee (Github: @suprateembanerjee)

import numpy as np
import cv2
from pathlib import Path
import time
import glob
import re

from occlusion_methods import occlude_welford, occlude_pixel_history, occlude_yolo
from yolopredictor import YoloPredictor
from localizers import mask_yolo

def occlude(source:str, occlusion_mechanism:str, PATH_TO_YOLO:str, weights:str='yolov7.pt') -> None:

	view_img = True 
	save_img = True
	vid_path = None
	save_separate = True

	localizer = 'yolo'
	occluder = 'welford'

	# Output save_path specifications

	save_path = Path('runs/occlude/exp')

	if not save_path.exists():
		save_dir = save_path #Path(str(save_path))
	else:
		dirs = glob.glob(f'{save_path}*')
		matches = [re.search(rf'%s(\d+)' % save_path.stem, d) for d in dirs]
		i = [int(m.groups()[0]) for m in matches if m]
		n = max(i) + 1 if i else 2
		save_dir = Path(f'{save_path}{n}')  

	(save_dir / 'labels').mkdir(parents=True, exist_ok=True)

	cap = cv2.VideoCapture(source)
	background_img = None
	background_update = None
	age_array = None
	history_array = None
	timeout = 30

	predictor = YoloPredictor(path_to_yolo=PATH_TO_YOLO, weights=weights)

	t0 = time.time()
	
	while cap.isOpened():

		ret, frame = cap.read()

		if not ret:
			break

		if background_img is None:
			background_img = frame.copy()
			age_array = np.zeros(frame.shape[:2])

		if localizer == 'yolo':
			det, img, timestamps = predictor.predict(frame)

		age_array += 1.
		t1 = t2 = t3 = t4 = t5 = t6 = t7 = t8 = time.time()
		mask = np.ones_like(age_array)
		image = frame.copy()
		s = ''
		save_path = str(save_dir / source.split('/')[-1])

		if occlusion_mechanism == 'yolo':

			display_windows = ['original', 'detection', 'background_img', 'background_update']

			# det, img, timestamps = predictor.predict(frame)
			t1 = timestamps['t1']
			t2 = timestamps['t2']
			t3 = timestamps['t3']

			occlude_out = occlude_yolo(det, s, img, predictor.names, predictor.colors, frame, mask, age_array, background_img, predictor.funcs)
			s, mask_out, background_img, background_update, age_array = occlude_out.values()

		if occlusion_mechanism == 'yolo welford':

			display_windows = ['original', 'mask', 'background_img', 'out_img']

			t1 = timestamps['t1']
			t2 = timestamps['t2']
			t3 = timestamps['t3']

			mask_local, s = mask_yolo(det, s, img, predictor.names, predictor.colors, frame, predictor.funcs)
			
			age_array *= np.logical_not(mask_local)

			mask_local_aged = np.logical_not((age_array > 20).astype(np.uint8) * np.logical_not(mask_local))

			occlude_out = occlude_welford(history_array, image, background_img)
			history_array, mask_background, background_img, _, timestamps = occlude_out.values()

			t4 = timestamps['t4']
			t5 = timestamps['t5']
			t6 = timestamps['t6']
			t7 = timestamps['t7']
			t8 = timestamps['t8']

			mask_out = np.logical_not(mask_local_aged) + (mask_local_aged * mask_background)
			mask_out[0][0] = 0. # Done to enable proper normalization

			out_img = background_img.copy()
			out_img[np.nonzero(mask_out)] = image[np.nonzero(mask_out)]

			mask_out = cv2.normalize(mask_out, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
			mask_out = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)
			

		elif occlusion_mechanism == 'per pixel history 200':

			display_windows = ['original', 'mask', 'background_img']

			occlude_out = occlude_pixel_history(timeout, history_array, image, background_img)
			history_array, mask_out, background_img, timestamps = occlude_out.values()
			t4 = timestamps['t4']
			t5 = timestamps['t5']
			t6 = timestamps['t6']
			t7 = timestamps['t7']
			t8 = timestamps['t8']
			timeout -= 1

			mask_out = cv2.normalize(mask_out, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
			mask_out = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)

		elif occlusion_mechanism == 'welford':

			display_windows = ['original', 'mask', 'background_img']

			occlude_out = occlude_welford(history_array, image, background_img)
			history_array, mask_out, background_img, _, timestamps = occlude_out.values()
			t4 = timestamps['t4']
			t5 = timestamps['t5']
			t6 = timestamps['t6']
			t7 = timestamps['t7']
			t8 = timestamps['t8']

			mask_out = cv2.normalize(mask_out, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
			mask_out = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)

		elif occlusion_mechanism == 'welford + erode + dilate':

			display_windows = ['original', 'mask', 'background_img', 'eroded+dilated mask']

			occlude_out = occlude_welford(history_array, image, background_img, erode_dilate=True)
			history_array, mask_out, background_img, eroded_dilated, timestamps = occlude_out.values()
			t4 = timestamps['t4']
			t5 = timestamps['t5']
			t6 = timestamps['t6']
			t7 = timestamps['t7']
			t8 = timestamps['t8']

			mask_out = cv2.normalize(mask_out, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
			mask_out = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)
			eroded_dilated = cv2.normalize(eroded_dilated, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
			eroded_dilated = cv2.cvtColor(eroded_dilated, cv2.COLOR_GRAY2BGR)

		# elif occlusion_mechanism == 'multimodal clustering':

		# 	display_windows = ['original', 'mask', 'background_img']

		# 	occlude_out = occlude_histogram()
			# history_array, mask_out, background_img, eroded_dilated, timestamps = occlude_out.values()
			# t4 = timestamps['t4']
			# t5 = timestamps['t5']
			# t6 = timestamps['t6']
			# t7 = timestamps['t7']
			# t8 = timestamps['t8']

		t9 = time.time()

		# Print time (inference + NMS)
		print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, \
({(1E3 * (t5 - t4)):.1f}ms) Step 1, ({(1E3 * (t6 - t5)):.1f}ms) Step 2, \
({(1E3 * (t7 - t6)):.1f}ms) Step 3, ({(1E3 * (t8 - t7)):.1f}ms) Step 4, \
({(1E3 * (t9 - t8)):.1f}ms) Step 5, ({(1E3 * (t9 - t1)):.1f}ms) Total')

		# Stream results
		if view_img:
			if 'detection' in display_windows:
				cv2.namedWindow('detection') 
				cv2.moveWindow('detection', 0, 0)
				cv2.imshow('detection', frame)

			if 'background_update' in display_windows:
				cv2.namedWindow('background update') 
				cv2.moveWindow('background update', background_update.shape[1], background_update.shape[0] + 30)
				cv2.imshow('background update', background_update)

			if 'out_img' in display_windows:
				cv2.namedWindow('output image') 
				cv2.moveWindow('output image', out_img.shape[1], out_img.shape[0] + 30)
				cv2.imshow('output image', out_img)
			
			if 'mask' in display_windows:
				cv2.namedWindow('mask') 
				cv2.moveWindow('mask', 0, 0)
				cv2.imshow('mask', mask_out)
			
			if 'original' in display_windows:
				cv2.namedWindow('original') 
				cv2.moveWindow('original', image.shape[1], 0)
				cv2.imshow('original', image)
			
			if 'background_img' in display_windows:
				cv2.namedWindow('background_image') 
				cv2.moveWindow('background_image', 0, background_img.shape[0] + 30)
				cv2.imshow('background_image', background_img)
			
			if 'eroded+dilated mask' in display_windows:    
				cv2.namedWindow('eroded+dilated mask') 
				cv2.moveWindow('eroded+dilated mask', eroded_dilated.shape[1], eroded_dilated.shape[0] + 30)
				cv2.imshow('eroded+dilated mask', eroded_dilated)

			cv2.waitKey(1)  # 1 millisecond

		# Save results (image with detections)
		if vid_path != save_path:  # new video
			vid_path = save_path
			vid_cap = cap
			if vid_cap:  # video
				fps = vid_cap.get(cv2.CAP_PROP_FPS)
				w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
				h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			else:  # stream
				fps, w, h = 30, im0.shape[1], im0.shape[0]
				save_path += '.mp4'
			if not save_separate:
				vid_writer =  cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
			else:
				if 'detection' in display_windows:
					vid_writer1 = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
				if 'background_update' in display_windows:
					vid_writer2 = cv2.VideoWriter(f'{save_path.split(".")[0]}_background_update.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
				if 'background_img' in display_windows:
					vid_writer3 = cv2.VideoWriter(f'{save_path.split(".")[0]}_background_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
				if 'mask' in display_windows:
					vid_writer4 = cv2.VideoWriter(f'{save_path.split(".")[0]}_mask.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
				if 'out_img' in display_windows:
					vid_writer5 = cv2.VideoWriter(f'{save_path.split(".")[0]}_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
				if 'eroded+dilated mask' in display_windows:
					vid_writer6 = cv2.VideoWriter(f'{save_path.split(".")[0]}_eroded_dilated_mask.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
			
		if not save_separate:
			compound = np.zeros(tuple(np.array(frame.shape[:-1]) * 2) + (3,), dtype=np.float32)
			h, w, _ = frame.shape
			compound[:h, :w, :] = mask_out if 'mask' in display_windows else frame
			if 'background_img' in display_windows:
				compound[h:, :w, :] = background_img
			compound[:h, w:, :] = image
			compound[h:, w:, :] = eroded_dilated if 'eroded+dilated mask' in display_windows else background_update
			compound = cv2.normalize(compound.copy(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
			vid_writer.write(compound)


		else:
			if 'detection' in display_windows:
				vid_writer1.write(frame)
			if 'background_update' in display_windows:
				vid_writer2.write(background_update)
			if 'background_img' in display_windows:
				vid_writer3.write(background_img)
			if 'mask' in display_windows:
				vid_writer4.write(mask_out)
			if 'out_img' in display_windows:
				vid_writer5.write(out_img)
			if 'eroded+dilated mask' in display_windows:
				vid_writer6.write(eroded_dilated)

	print(f'Done. ({time.time() - t0:.3f}s)')

	cap.release()

if __name__=='__main__':

	file = ['outdoor.mp4', 
			'oslo.mp4'][0]
	occlusion_mechanism = ['yolo',
						'yolo welford',
						'per pixel history 200',
						'welford',
						'welford + erode + dilate',
						'multimodal clustering'][1]

	PATH_TO_YOLO = '/Users/suprateembanerjee/Python Projects/Teleport/Occlude/YOLO/yolov7-main'
	weights=['yolov7.pt',
			 'yolov7-w6.pt',
			 'yolov7x.pt'][0]

	occlude(f'res/{file}', occlusion_mechanism, PATH_TO_YOLO, weights)