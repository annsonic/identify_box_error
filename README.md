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

And this reference
https://medium.com/data-science-at-microsoft/error-analysis-for-object-detection-models-338cb6534051

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

## Execution
```
python main.py 
--yaml_path <the data.yaml used by yolov8> 
--data_subset <train, val or test> 
--predict_folder_path <the output folder used by yolov8>
--analysis_folder_path <the output folder used by this project>
```
Ex. 
yaml_path = "datasets/cfg/coco128.yaml"
data_subset = "tarin"
predict_folder_path = "/run/detect/"
analysis_folder_path = "predictions/analyzed_train"
