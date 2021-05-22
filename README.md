## TensorRT Implementation

### Introduction
This source makes use of TensorRT engine by Nvidia and has a reported fps of 19. The model engine has eliminated wpod-net (as it was causing a lot of lag) and has deployed one single architecture for both vehicle detection and license plate detection

### Easter egg
Because of the video stream access, for the output results, the use of time redundancy is for the same vehicle combined with multiple frames to fuse the results to improve the output accuracy

### Requirements
- TensorRT 7.0.0.11
- numpy    1.14
- onnx     1.6.0
- opencv-python 4.2.0.32

### Usage
* Build Darknet framework
  ```shellscript
  $ cd darknet && make
  ```
* Convert Keras model(json&h5) to onnx
  ```shellscript
  $ python h52onnx.py
  ```
* Run inference.py (look out for optional arguments)
  ```shellscript
  $ python inference.py --a demo.mp4
  ```
* Run flask application
```shellscript
$ python app.py
```

### References
* [shuangyichen](https://github.com/shuangyichen/alpr-TensorRT7)'s implementation