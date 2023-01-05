from occlusion_methods import occlude_welford, occlude_pixel_history
from yolo_utils import YoloPredictor

import numpy as np
import cv2
from pathlib import Path
import time
import glob
import re

def occlude(source:str):

	view_img = True 
	save_img = True
	vid_path = None
	save_separate = True

	# Output path specifications

	path = Path('runs/detect/exp')

	if not path.exists():
		save_dir = path #Path(str(path))
	else:
		dirs = glob.glob(f'{path}*')
		matches = [re.search(rf'%s(\d+)' % path.stem, d) for d in dirs]
		i = [int(m.groups()[0]) for m in matches if m]
		n = max(i) + 1 if i else 2
		save_dir = Path(f'{path}{n}')  

	(save_dir / 'labels').mkdir(parents=True, exist_ok=True)



	cap = cv2.VideoCapture(source)
	background_img = None
	background_update = None
	age_array = None
	history_array = None
	timeout = 30

	occlusion_mechanism = ['yolo',
							'per pixel history 200',
							'welford',
							'welford + erode + dilate'][0]

	if occlusion_mechanism == 'yolo':
		predictor = YoloPredictor()
	
	while cap.isOpened():

		ret, frame = cap.read()
		if ret == False:
			break

		if background_img is None:
			background_img = frame.copy()
			age_array = np.zeros(frame.shape[:2])

		age_array += 1.
		t1 = t2 = t3 = t4 = t5 = t6 = t7 = t8 = time.time()
		mask = np.ones_like(age_array)
		image = frame.copy()
		s = ''
		save_path = str(save_dir / source)

		if occlusion_mechanism == 'yolo':

			display_windows = ['original', 'detection', 'background_img', 'background_update']

			det, img, mask = predictor.predict(frame)
			occlude_out = predictor.occlude_yolo(det, s, img, frame, mask)
			s, mask_out, background_img, background_update = occlude_out.values()

		elif occlusion_mechanism == 'per pixel history 200':

			display_windows = ['original', 'mask', 'background_img']

			occlude_out = occlude_pixel_history(timeout, history_array, image, background_img)
			history_array, mask_out, background_img = occlude_out.values()
			timeout -= 1

		elif occlusion_mechanism == 'welford':

			display_windows = ['original', 'mask', 'background_img']

			occlude_out = occlude_welford(history_array, image, background_img)
			history_array, mask_out, background_img, _ = occlude_out.values()

		elif occlusion_mechanism == 'welford + erode + dilate':

			display_windows = ['original', 'mask', 'background_img', 'eroded+dilated mask']

			occlude_out = occlude_welford(history_array, image, background_img, erode_dilate=True)
			history_array, mask_out, background_img, eroded_dilated = occlude_out.values()

		t9 = time.time()

		# Print time (inference + NMS)
		print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, ({(1E3 * (t5 - t4)):.1f}ms) Step 1, \
({(1E3 * (t6 - t5)):.1f}ms) Step 2, ({(1E3 * (t7 - t6)):.1f}ms) Step 3, ({(1E3 * (t8 - t7)):.1f}ms) Step 4, \
({(1E3 * (t9 - t8)):.1f}ms) Step 5, ({(1E3 * (t9 - t4)):.1f}ms) Total')

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
			# if isinstance(vid_writer, cv2.VideoWriter):
			# 	vid_writer.release()  # release previous video writer
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
				if 'eroded+dilated mask' in display_windows:
					vid_writer5 = cv2.VideoWriter(f'{save_path.split(".")[0]}_eroded_dilated_mask.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
			
		if not save_separate:
			compound = np.zeros(tuple(np.array(frame.shape[:-1]) * 2) + (3,), dtype=np.float32)
			h, w, _ = frame.shape
			compound[:h, :w, :] = mask_out if 'mask' in display_windows else frame
			if 'background_img' in display_windows:
				compound[h:, :w, :] = background_img
			compound[:h, w:, :] = image
			compound[h:, w:, :] = eroded_dilated if 'eroded+dilated mask' in display_windows else background_update
			compound = cv2.normalize(compound.copy(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
			print(compound.shape)
			vid_writer.write(compound)


		else:
			if 'detection' in display_windows:
				# print(f'im0: {type(im0)} {im0.shape}')
				vid_writer1.write(frame)
			if 'background_update' in display_windows:
				# print(f'background_update: {type(background_update)} {type(background_update[0][0][0])} {background_update.shape}')
				vid_writer2.write(background_update)
			if 'background_img' in display_windows:
				# print(f'background_img: {type(background_img)} {type(background_img[0][0][0])} {background_img.shape}')
				vid_writer3.write(background_img)
			if 'mask' in display_windows:
				# print(f'mask: {type(mask_out)} {type(mask_out[0][0])} {mask_out.shape}')
				vid_writer4.write(mask_out)
			if 'eroded+dilated mask' in display_windows:
				# print(f'eroded_dilated: {type(eroded_dilated)} {type(eroded_dilated[0][0])} {eroded_dilated.shape}')
				vid_writer5.write(eroded_dilated)

	# if save_txt or save_img:
	# 	s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
	# 	#print(f"Results saved to {save_dir}{s}")

	print(f'Done. ({time.time() - t0:.3f}s)')

	cap.release()

if __name__=='__main__':
	occlude('outdoor.mp4') # 'oslo.mp4' 'outdoor.mp4' 'people.mp4'