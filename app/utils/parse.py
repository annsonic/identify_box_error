import gc
from pathlib import Path
import re
from typing import Generator, Union
import warnings

import numpy as np
import yaml

from app.utils import (ROOT, DATASET_CACHE_VERSION)
from app.utils.ops import xywh2xyxy, sort_counter_clockwise_in_numpy_coord


def check_class_names(names: Union[list, dict]) -> dict:
    """
    Check class names.
    Map imagenet class codes to human-readable names if required.
    Convert lists to dicts.
    Source: ultralytics/nn/autobackend.py
    """
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):  # imagenet class codes, i.e. 'n01440764'
            names_map = yaml_load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # human-readable names
            names = {k: names_map[v] for k, v in names.items()}
    return names


def safe_load_yaml(file: str = "data.yaml") -> dict:
    """ Read the content of a YAML file and return as a dict. """
    assert Path(file).suffix in (".yaml", ".yml"), f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string
        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
    return data


def yaml_load(file: str = "data.yaml") -> dict:
    """
    1. Read the content of a YAML file
    2. Convert class name mapping
    3. get the absolute folder path for train/val/test sets.
    """
    data = safe_load_yaml(file)

    data["names"] = check_class_names(data["names"])
    data['yaml_file'] = Path(file)
    if not Path(file).is_absolute():
        data['yaml_file'] = (ROOT / file).resolve()
    folder = data['yaml_file'].parent
    for k in "train", "val", "test":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (folder / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (folder / data[k][3:]).resolve()
                data[k] = x
            else:
                data[k] = [(folder / x).resolve() for x in data[k]]

    return data


def load_dataset_cache_file(path: Path) -> dict:
    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict and convert to python float
    gc.enable()
    return cache


def cache_load(subset: str, yaml_data: dict, is_obb: bool = False) -> list[dict]:
    """ Load dataset cache file according to the yaml_data.
    Args:
        subset (str): range['train', 'val', 'test']
        yaml_data (dict): accepts the output of the function `yaml_load`
        is_obb (bool): if True, load the segments instead of boxes for oriented bounding boxes.
    Returns:
        list of dict

    The content of each row in the list is:
    {
        'bbox_format': 'xyxy',
        'bboxes': ndarray(n, 4) in float
        'cls': ndarray(n, ) in int
        'im_file': str -> absolute path
        'normalized': True
        'shape': tuple(int, int) -> (height, width)
        'segments': list of ndarray(4, 2) in float. The vertices are arranged in a counter-clockwise order.
    }
    n is the number of bounding boxes in the image.
    """
    cache_path = Path(str(yaml_data[subset]).replace('images', 'labels')).with_suffix(".cache")
    try:
        cache = load_dataset_cache_file(cache_path)
        assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
    except (FileNotFoundError, AssertionError, AttributeError):
        raise Exception(f"Please check the cache file {cache_path}.")
    # Read cache
    [cache.pop(k) for k in ("hash", "version", "msgs") if k in cache.keys()]  # remove items
    for data in cache["labels"]:
        data['bbox_format'] = "xyxy"
        data['cls'] = data['cls'].flatten().astype(int)
        if is_obb:
            data['segments'] = [sort_counter_clockwise_in_numpy_coord(box) for box in data['segments']]
        else:
            data['bboxes'] = xywh2xyxy(data['bboxes']) if data['bboxes'].shape[0] > 0 else np.empty((0,))
    return cache["labels"]


def load_one_label_file(file: str, has_conf=False
                        ) -> Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """ Given a text file, parse the content to a list of xywh-box coordinates.

    Args:
        file (str): path of the text file.
        has_conf (bool): if True, read the `conf` confidence score.
    Return:
        classes (ndarray): ndarray(n, ) in int
        boxes (ndarray): ndarray(n, 4) or (n, 8) in float
        confs (ndarray): ndarray(n, 1) in float
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = np.loadtxt(file, ndmin=2)
    # Reshape to 2D if only one row
    if data.ndim == 1:
        data = data.reshape(1, -1)
    classes, boxes, confs = np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))
    if data.size != 0:
        classes = data[:, 0].astype(int)
        if has_conf:
            boxes = data[:, 1:-1]
            confs = data[:, -1]
        else:
            boxes = data[:, 1:]
    if has_conf:
        return classes, boxes, confs
    return classes, boxes


def txt_load(folder: str, has_conf=False, is_obb: bool = False) -> dict[str, dict]:
    """ Given a folder of text files, parse the content to a list of dict.
    Args:
        folder (str): path of the label folder.
        has_conf (bool): if True, read the `conf` confidence score.
        is_obb (bool): if True, load the segments instead of boxes for oriented bounding boxes.
    Returns:
        nested dict
            The content of each row in the dict is:
            filename: {
                'bbox_format': 'xyxy',
                'bboxes': ndarray(n, 4) in float
                'cls': ndarray(n, ) in int
                'normalized': True
                'segments': list of ndarray(4, 2) in float. The vertices are arranged in a counter-clockwise order.
            }
    n is the number of bounding boxes in the image.
    """
    files = list(Path(folder).rglob("*.txt"))
    labels = dict()
    for file in files:
        labels[str(file.name)] = dict()
        if has_conf:
            cls, bboxes, confs = load_one_label_file(str(file), has_conf)
            labels[str(file.name)]["conf"] = confs
        else:
            cls, bboxes = load_one_label_file(str(file), has_conf)
        if is_obb:
            labels[str(file.name)] |= {
                "bboxes": np.empty((0,)),
                "segments": [sort_counter_clockwise_in_numpy_coord(box) for box in bboxes.reshape(-1, 4, 2)],
            }
        else:
            labels[str(file.name)] |= {
                "bboxes": xywh2xyxy(bboxes) if bboxes.shape[0] > 0 else np.empty((0,)),
            }
        labels[str(file.name)] |= {
            "cls": cls,
            "normalized": True,
            "bbox_format": "xyxy",
            "bad_box_errors": [],  # list item: type of error
            "missing_box_errors": [],  # list item: index of ground truth boxes
        }

    return labels


def truth_prediction_fetcher(yaml_path: str, subset: str, predict_folder_path: str, is_obb: bool = False
                             ) -> Generator[tuple[dict, dict], None, None]:
    """
    Args:
        yaml_path (str): path of the yaml file.
        subset (str): range['train', 'val', 'test']
        predict_folder_path (str): path of the prediction folder which stores the inference text files.
        is_obb (bool): if True, load the segments instead of boxes for oriented bounding boxes.
    Returns:
        tuple(dict, dict): (ground truth, prediction)
    """
    yaml_data = yaml_load(yaml_path)
    facts = cache_load(subset, yaml_data, is_obb=is_obb)
    guesses = txt_load(predict_folder_path, has_conf=True, is_obb=is_obb)
    assert len(facts) == len(guesses), f"Number of files in {subset} and predictions do not match."

    for fact in facts:
        txt_file_name = Path(fact['im_file']).stem + ".txt"
        guess = guesses[txt_file_name]
        yield fact, guess
