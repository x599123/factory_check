#!/bin/bash
python test_4-videogear.py&
/home/ubuntu/anaconda3/envs/patrol/bin/python live-videogear.py &
/home/ubuntu/anaconda3/envs/yolov5/bin/python digit_combine-videogear.py &
/home/ubuntu/anaconda3/envs/yolov5/bin/python handle-videogear.py &

