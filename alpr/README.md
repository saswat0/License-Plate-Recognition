### Setup Instructions

* Download the model weights
```shellscript
$ bash download.sh
```
* Configure darknet
```shellscript
$ cd darknet && make
```

### Requirements

* Ubuntu 16.04
* Keras 2.2.4
* TensorFlow 1.5.0
* OpenCV 2.4.9
* NumPy 1.14
* Python 2.7

Install requirements:
```bash
$ pip install -r requirements.txt
```

### Running on example video
*   Download the video file from [here](http://youtube.com/watch?v=hv94fk7ldS8) and save it as demo.mp4
*   **Without vehicle detection** (Faster but picks up any text on the frame)
```shellscript
$ python video_support_v1.py
```
*   **With vehicle detection** (Slower but picks up text from detected vehicles only)
```shellscript
$ python video_support_v2.py
```
* **Latest Code**
```bash
$ python video.py
```

### References

* Paper webpage: http://sergiomsilva.com/pubs/alpr-unconstrained/

If you use results produced by our code in any publication, please cite our paper:

```
@INPROCEEDINGS{silva2018a,
  author={S. M. Silva and C. R. Jung}, 
  booktitle={2018 European Conference on Computer Vision (ECCV)}, 
  title={License Plate Detection and Recognition in Unconstrained Scenarios}, 
  year={2018}, 
  pages={580-596}, 
  doi={10.1007/978-3-030-01258-8_36}, 
  month={Sep},}
```