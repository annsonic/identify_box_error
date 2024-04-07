from PIL import Image
import numpy as np
from pathlib import Path
import shutil
import yaml

import pytest

from app.utils import DATASET_CACHE_VERSION
from app.utils.parse import load_one_label_file

TMP = Path(__file__).resolve().parent / "tmp"  # temp directory for test files
IMG_SHAPE = (100, 100)


class TextFilePreparer:
    def __init__(self, label_type: str = "predicted"):
        self.label_type = label_type
        if label_type == "ground_truth":
            self.folder = TMP / "ground_truth" / "labels" / "test"
        else:
            self.folder = TMP / "predicted" / "labels"
        self.folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def check_cls_x_y_w_h_range(content: tuple):
        if len(content) == 5:
            (cls, x, y, w, h) = content
        else:
            (cls, x, y, w, h, score) = content
            assert 0 <= score <= 1
        assert cls >= 0
        assert 0 <= x <= 1
        assert 0 <= y <= 1
        assert 0 <= w <= 1
        assert 0 <= h <= 1

    def create(self, file_name: str, contents: list[tuple]):
        """ Write the content into a text file. """
        with open(self.folder / file_name, "w") as f:
            for content in contents:
                TextFilePreparer.check_cls_x_y_w_h_range(content)
                f.write(" ".join(map(str, content)) + "\n")


class FakeData:
    """
    three_boxes: [(0.5, 0.5, 0.5, 0.52), (0.42, 0.4, 0.32, 0.48), (0.6, 0.55, 0.34, 0.34)]
    ┌───────┐
    │ ▂▂▂   │
    │ ███▄▄ │
    │   ███ │
    └───────┘
    two_boxes_positive: [(0.35, 0.52, 0.5, 0.2), (0.5, 0.55, 0.56, 0.18)]
     ████▆
     two_boxes_negative: [(0.5, 0.55, 0.6, 0.5), (0.35, 0.4, 0.5, 0.2)]
     ████▆
      ████

    one_box: [(0.8, 0.7, 0.2, 0.3)]

    test_cases:
    1: empty
    2: 1 true positive, 1 background, 1 bad location (iou=0.448, 0.407)
    3: 1 wrong class, 2 missing
    4: 1 wrong class location, 1 true positive, 1 duplicate
    5: 1 true positive (iou=1)
    6: 1 false positive (iou=0.25) => bad location
    7: 1 false positive => background
    """
    def __init__(self):
        self.num_images = 5
        self.yaml_path = TMP / "insects.yaml"
        self.img_folder_path = TMP / "ground_truth" / "images" / "test"
        self.img_folder_path.mkdir(parents=True, exist_ok=True)
        self.txt_folder_path = TMP / "ground_truth" / "labels" / "test"
        self.txt_folder_path.mkdir(parents=True, exist_ok=True)
        self.predict_folder_path = TMP / "predicted" / "labels"
        self.predict_folder_path.mkdir(parents=True, exist_ok=True)
        self.cache_path = TMP / "ground_truth" / "labels" / "test.cache"

        self.prepare_yaml()
        self.prepare_txt()
        self.prepare_image()
        self.prepare_cache()

    def prepare_yaml(self):
        json_data = {
            "test": str(self.img_folder_path),
            "names": {
                "0": "butterfly",
                "1": "moth",
            },
        }
        yaml_path = self.yaml_path
        with open(yaml_path, "w") as f:
            yaml.safe_dump(json_data, f)

    def prepare_txt(self):
        truth_preparer = TextFilePreparer('ground_truth')
        guess_preparer = TextFilePreparer('predicted')

        truth_preparer.create('1.txt', [])
        guess_preparer.create('1.txt', [])
        truth_preparer.create('2.txt', [(0, 0.3, 0.4, 0.2, 0.3), (0, 0.6, 0.55, 0.34, 0.34), (1, 0.6, 0.8, 0.4, 0.26)])
        guess_preparer.create('2.txt', [(0, 0.26, 0.6, 0.18, 0.24, 0.881), (0, 0.7, 0.3, 0.2, 0.4, 0.799), (1, 0.6, 0.8, 0.4, 0.26, 0.697)])
        truth_preparer.create('3.txt', [(0, 0.4, 0.3, 0.2, 0.2), (0, 0.4, 0.6, 0.2, 0.2), (0, 0.7, 0.4, 0.2, 0.2)])
        guess_preparer.create('3.txt', [(1, 0.4, 0.3, 0.2, 0.2, 0.973)])
        truth_preparer.create('4.txt', [(0, 0.3, 0.4, 0.2, 0.3), (0, 0.7, 0.4, 0.2, 0.2)])
        guess_preparer.create('4.txt', [(1, 0.26, 0.6, 0.18, 0.24, 0.881), (0, 0.7, 0.4, 0.2, 0.2, 0.852), (0, 0.7, 0.4, 0.2, 0.2, 0.335)])
        truth_preparer.create('5.txt', [(0, 0.5, 0.4, 0.3, 0.3), (1, 0.4, 0.7, 0.2, 0.2), (1, 0.62, 0.84, 0.2, 0.2)])
        guess_preparer.create('5.txt', [(0, 0.42, 0.44, 0.28, 0.24, 0.968), (0, 0.7, 0.4, 0.2, 0.2, 0.567), (1, 0.5, 0.76, 0.5, 0.3, 0.685)])

    def prepare_image(self):
        # Create blank images
        for i in range(1, self.num_images + 1):
            im = Image.new("RGB", IMG_SHAPE, color='black')
            im.save(self.img_folder_path / f"{i}.jpg")

    def prepare_cache(self):
        img_files = sorted(list(self.img_folder_path.glob("*.jpg")))
        label_files = sorted(list(self.txt_folder_path.glob("*.txt")))
        x = {"labels": []}
        for img, label in zip(img_files, label_files):
            classes, boxes = load_one_label_file(str(label))
            x["labels"].append(
                dict(
                    im_file=str(img),
                    shape=IMG_SHAPE,
                    cls=classes,  # n, 1
                    bboxes=boxes,  # n, 4
                    normalized=True,
                    bbox_format="xywh",
                )
            )
        x["version"] = DATASET_CACHE_VERSION
        np.save(str(self.cache_path), x)
        self.cache_path.with_suffix(".cache.npy").rename(self.cache_path)  # remove .npy suffix


@pytest.fixture
def session_setup():
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    fake = FakeData()
    pytest.yaml_path = fake.yaml_path
    pytest.predict_folder_path = fake.predict_folder_path
