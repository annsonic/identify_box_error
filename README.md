# identify_box_error
Identifying Object Detection Errors for YOLO model

## Motivation
Inspired by the paper,
@inproceedings{tide-eccv2020,
  author    = {Daniel Bolya and Sean Foley and James Hays and Judy Hoffman},
  title     = {TIDE: A General Toolbox for Identifying Object Detection Errors},
  booktitle = {ECCV},
  year      = {2020},
}

## Installation
```
$ conda create --name box_analyze python=3.12
$ conda activate box_analyze
$ pip install poetry
$ cd <this_repo>
$ poetry install
```

## Unit test
```
python -m pytest test
```
