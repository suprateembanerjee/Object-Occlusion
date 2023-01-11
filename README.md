# Object Occlusion
This project is about occluding people from video scenes. Many of the methods have simply explored background imaging using various techniques. Inferring the background of a scene lets us protect privacy of residents in a video stream from a private property, for instance. In my case, it needed to be operable in real time, so that this stack can run on a live video feed on the cloud before being served to a client.

# How to Run

We access the system from `occlude.py`.

Four parameters need to be specified: 
* `file`: Name of the input video file, placed under the `res` folder.
* `occlusion_mechanism`: Method to be used to occlude. 
* `path to YOLO`: Absolute path to YOLOv7 (parent directory of the [Official GitHub Repo](https://github.com/WongKinYiu/yolov7)), with desired weights downloaded and placed inside it (Eg [yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)).
* `weights` Name of the file containing weights to initialize the model, Eg. 'yolov7.pt'.

These parameters are specified directly inside the `occlude.py` (lines 216:226) instead of using standard python argument parsing for convenience.

Once accurately specified, run the following command from the parent directory as: 

```python src/occlude.py```

# Comparison and Analysis of Methods

Four metrics are considered while judging the applicability of methods in this scenario:
* Speed (what is the total time needed to operate on one frame?)
* Artifacts (can a reasonably perceptible human tell between an occluded stream and an original background stream?)
* Convergence (does the method converge to produce an accurate background image over time?)
* Localization (does the method enable us to be specific on what to occlude?)

The following methods have been implemented:
* YOLOv7 based

  This method utilizes YOLOv7 object detection (which is performant as well as reasonably accurate) to detect individuals and mask them using the latest background image. Not considering the detection mechanism by YOLOv7, this method is extremely simple. We initialize the background image with the first frame received, then run detections. When a person is detected, it "cuts out" the bounding box from the latest background image, and replaces the pixels in the frame with the cut-out. This method results in artifacts like shadows and out-of-box parts of the person caused by sudden limb movement, and such. This method converges well due to the fact that it does not impose too much weight on initial conditions (initial background image from first frame). The background updates for every pixel where a detection has not been made in the last 20 frames. This mechanism helps with dropped detections by YOLO within a series of frames. This method can be used to localize well, because detections by YOLO allows us to disregard occlusion of other elements such as cars and animals, for instance.
  
* Pure Background Imaging

  The following methods do not localize, and are attempts at inferring the background of a scene.
  
  * Historical Distribution per channel per pixel
    
    Every pixel consists of 3 channels in the RGB colorspace. To estimate the background, we look at the past 200 frames, and store the values for each pixel in an array shaped `(H, W, C, min(n, 200))`. We compute the mean and standard deviation of every pixel for every frame. If in a new frame, the new `pixel[channel]` value encountered is under a certain standard deviation threshold, we add the new frame's `pixel[channel]` to our distribution, popping the oldest of the last 200 frames. This operation scales very poorly in terms of performance, as the distribution array involves a lot of compute (for appending, popping, calculating mean and std) at every frame. It can take up to 2000ms for a frame (on a 200 frame pipe) on a M1 Max system. This method also has a fairly strong prior set by the initial background image, and converges poorly. However, this method is free of artifacts if the initial background image is devoid of subjects.
    
  * Welford's Algorithm to update Historical Distribution per channel per pixel
    
    One of the major problems with the previous method was performance. The 200-frame pipe of values per channel was too much information to store and compute on. We used Welford's online updation algorithm to infer mean and standard deviation per channel at every timestamp, eliminating the need for storing frames. The history array for this method has a shape `(H, W, C, 4)`, where we store `mean`, `std`, `s`, and `num_frames_encountered_before` in the last dimension. This is very performant, and offers about 27x the speed of the previous method, taking an average of 75ms per frame on the same footage. The rest of the operation is similar, where we threshold using standard deviation from the online mean to update the background image. As a result, this faces the same convergence issues but contains minimal artifacts.
    
    The mask created on a per-pixel basis seems to be noisy. To help this, the mask was passed through an erosion-dilation-erosion pipeline. This method changed the mask that was being used, but did not seem to have any reasonable impact on the final outcome (background image).
