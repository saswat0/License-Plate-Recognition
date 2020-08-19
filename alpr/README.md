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


### Running on example video
*   Download the video file from [here](http://youtube.com/watch?v=hv94fk7ldS8) and save it as demo.mp4
*   Run the code by
```bash
$ python video.py
```
* Run the flask app on different terminal
```
$ python app.py
```
* Open [0.0.0.0:5000](0.0.0.0:5000) in your web browser

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