### Setup Instructions

* Download the model weights
```shellscript
$ bash get-networks.sh
```
* Configure darknet
```shellscript
$ cd darknet && make
```

Install requirements:
```bash
$ pip install -r requirements.txt
```

### Running a simple test on given images

```shellscript
$ bash run.sh -i samples/test -o /tmp/output -c /tmp/output/results.csv
```

### To test the lastest code
*   Download the video file from [here](http://youtube.com/watch?v=hv94fk7ldS8) and save it as demo.mp4
*   Run the base detection code
```bash
python video.py
```
*   Run flask app
```bash
python app.py
```