import os
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import pytest
import shutil
import yaml

from app.utils import DATASET_CACHE_VERSION
from app.utils.parse import load_one_label_file

TMP = Path(__file__).resolve().parent / "tmp"  # temp directory for test files
IMG_SHAPE = (400, 400)


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
        cls = content[0]
        assert cls >= 0
        for value in content[1:]:
            assert 0 <= value <= 1

    def create(self, file_name: str, contents: list[tuple]):
        """ Write the content into a text file. """
        with open(self.folder / file_name, "w") as f:
            for content in contents:
                TextFilePreparer.check_cls_x_y_w_h_range(content)
                f.write(" ".join(map(str, content)) + "\n")


class FakeRegularBoxData:
    def __init__(self, is_rotated: bool = False):
        self.num_images = 0  # the value will be updated in prepare_txt_img
        self.is_rotated = is_rotated
        self.yaml_path = TMP / "insects.yaml"
        self.img_folder_path = TMP / "ground_truth" / "images" / "test"
        self.img_folder_path.mkdir(parents=True, exist_ok=True)
        self.txt_folder_path = TMP / "ground_truth" / "labels" / "test"
        self.txt_folder_path.mkdir(parents=True, exist_ok=True)
        self.predict_folder_path = TMP / "predicted" / "labels"
        self.predict_folder_path.mkdir(parents=True, exist_ok=True)
        self.cache_path = TMP / "ground_truth" / "labels" / "test.cache"

        self.prepare_yaml()
        self.prepare_txt_img()
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

    def prepare_txt_img(self):
        truth_preparer = TextFilePreparer('ground_truth')
        guess_preparer = TextFilePreparer('predicted')
        """
            test_cases:
            1: empty
            2: 1 background, 1 bad location, 1 true positive (iou=0.033, 0.116, 1.0)
            3: 1 wrong class, 2 missing (iou=1.0, 0.0, 0.0)
            4: 1 wrong class location, 1 true positive, 1 duplicate (iou=0.113, 1.0, 0.818)
            5: 1 background, 1 duplicate
            6: 2 backgrounds
            7: 1 true positive, 1 background, 1 bad location (iou=0.444, 0.08, 0.377)
        """
        data = [
            ('1.txt', [], []),
            ('2.txt', [(0, 0.3, 0.4, 0.2, 0.3), (0, 0.6, 0.55, 0.34, 0.34), (1, 0.6, 0.8, 0.4, 0.26)], 
                [(0, 0.22, 0.64, 0.18, 0.24, 0.881), (0, 0.7, 0.3, 0.2, 0.4, 0.799), (1, 0.6, 0.8, 0.4, 0.26, 0.697)]),
            ('3.txt', [(0, 0.4, 0.3, 0.2, 0.2), (0, 0.4, 0.6, 0.2, 0.2), (0, 0.7, 0.4, 0.2, 0.2)], 
                [(1, 0.4, 0.3, 0.2, 0.2, 0.973)]),
            ('4.txt', [(0, 0.3, 0.4, 0.2, 0.3), (0, 0.7, 0.4, 0.2, 0.2)], 
                [(1, 0.26, 0.6, 0.18, 0.24, 0.881), (0, 0.7, 0.4, 0.2, 0.2, 0.852), (0, 0.68, 0.4, 0.2, 0.2, 0.835)]),
            ('5.txt', [], [(0, 0.7, 0.4, 0.2, 0.2, 0.852), (0, 0.68, 0.4, 0.2, 0.2, 0.835)]),
            ('6.txt', [], [(0, 0.7, 0.4, 0.2, 0.2, 0.852), (1, 0.68, 0.4, 0.2, 0.2, 0.835)]),
            ('7.txt', [(0, 0.5, 0.4, 0.3, 0.3), (1, 0.4, 0.74, 0.2, 0.3), (1, 0.62, 0.84, 0.2, 0.3)], 
                [(0, 0.42, 0.44, 0.28, 0.24, 0.968), (0, 0.7, 0.4, 0.2, 0.2, 0.567), (1, 0.5, 0.82, 0.38, 0.3, 0.685)]),
        ]
        self.num_images = len(data)

        for (txt_name, ground, guess) in data:
            truth_preparer.create(txt_name, ground)
            guess_preparer.create(txt_name, guess)
            self.prepare_image(txt_name.replace('.txt', '.jpg'), ground, guess)

    def prepare_image(self, file_name: str, ground: list[tuple], guess: list[tuple]):
        def rectangle(box: tuple[float, float, float, float]):
            x, y, w, h = box
            x *= IMG_SHAPE[0]
            y *= IMG_SHAPE[1]
            w *= IMG_SHAPE[0]
            h *= IMG_SHAPE[1]
            dw = w / 2.0
            dh = h / 2.0
            return (x - dw), y - dh, x + dw, y + dh
            
        # Create blank images
        im = Image.new("RGB", IMG_SHAPE, color='black')
        # Create rectangle image 
        painter = ImageDraw.Draw(im)
        for cxywh in ground:
            color = 'Blue' if cxywh[0] == 0 else 'White'
            left, top, right, bottom = rectangle(cxywh[1:])
            painter.rectangle((left, top, right, bottom), outline=color, width=6)
        for cxywhf in guess:
            color = 'Yellow' if cxywhf[0] == 0 else 'Green'
            left, top, right, bottom = rectangle(cxywhf[1:5])
            painter.rectangle((left, top, right, bottom), outline=color, width=2)
        
        im.save(self.img_folder_path / file_name)

    def prepare_cache(self):
        img_files = sorted(list(self.img_folder_path.glob("*.jpg")))
        label_files = sorted(list(self.txt_folder_path.glob("*.txt")))
        x = {"labels": []}
        for img, label in zip(img_files, label_files):
            classes, boxes = load_one_label_file(str(label), has_conf=False)
            if self.is_rotated:
                # Reshape the (n, 8) to (n, 4, 2)
                boxes = boxes.reshape(-1, 4, 2)
                segments = [box for box in boxes]
                x["labels"].append(
                    dict(
                        im_file=str(img),
                        shape=IMG_SHAPE,
                        cls=classes,  # n, 1
                        bboxes=np.empty((0, 1)),
                        segments=segments,  # n, 4, 2
                        normalized=True,
                        bbox_format="xywh",
                    )
                )
            else:
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


class FakeRotatedBoxData(FakeRegularBoxData):
    def __init__(self, is_rotated: bool = True):
        super().__init__(is_rotated=is_rotated)

    def prepare_txt_img(self):
        truth_preparer = TextFilePreparer('ground_truth')
        guess_preparer = TextFilePreparer('predicted')
        """
            test_cases:
            1: empty
            2: 1 background, 1 bad location, 1 true positive (iou=0.033, 0.116, 1.0)
            3: 1 wrong class, 2 missing (iou=1.0, 0.0, 0.0)
            4: 1 wrong class location, 1 true positive, 1 duplicate (iou=0.113, 1.0, 0.818)
            5: 1 background, 1 duplicate
            6: 2 backgrounds
            7: 1 true positive, 1 background, 1 bad location (iou=0.444, 0.08, 0.377)
        """
        data = [
            ('1.txt', [], []),
            ('2.txt', [(0, 0.2675, 0.1975, 0.465, 0.2325, 0.4125, 0.5275, 0.2175, 0.4925),
                       (0, 0.47, 0.365, 0.8075, 0.425, 0.7475, 0.76, 0.41, 0.7),
                       (1, 0.39, 0.645, 0.785, 0.7125, 0.7375, 0.9725, 0.345, 0.9025)],
             [(0, 0.1525, 0.4525, 0.33, 0.4825, 0.2875, 0.72, 0.11, 0.6875, 0.881),
              (0, 0.6875, 0.12, 0.885, 0.155, 0.815, 0.55, 0.6175, 0.515, 0.799),
              (1, 0.39, 0.645, 0.785, 0.7125, 0.7375, 0.9725, 0.345, 0.9025, 0.697)]),
            ('3.txt', [(0, 0.26, 0.2475, 0.4575, 0.2825, 0.4225, 0.48, 0.225, 0.445),
                       (0, 0.2075, 0.5425, 0.405, 0.5775, 0.37, 0.775, 0.1725, 0.74),
                       (0, 0.5375, 0.3975, 0.735, 0.4325, 0.7, 0.63, 0.5025, 0.595)],
             [(1, 0.26, 0.2475, 0.4575, 0.2825, 0.4225, 0.48, 0.225, 0.445, 0.973)]),
            ('4.txt', [(0, 0.1525, 0.28, 0.35, 0.315, 0.2975, 0.61, 0.1, 0.575),
                       (0, 0.5375, 0.3975, 0.735, 0.4325, 0.7, 0.63, 0.5025, 0.595)],
             [(1, 0.0825, 0.5, 0.26, 0.5325, 0.2175, 0.7675, 0.04, 0.7375, 0.881),
              (0, 0.5375, 0.3975, 0.735, 0.4325, 0.7, 0.63, 0.5025, 0.595, 0.852),
              (0, 0.5175, 0.395, 0.715, 0.43, 0.68, 0.6275, 0.4825, 0.5925, 0.835)]),
            ('5.txt', [],
             [(0, 0.5375, 0.3975, 0.735, 0.4325, 0.7, 0.63, 0.5025, 0.595, 0.852),
              (0, 0.5175, 0.395, 0.715, 0.43, 0.68, 0.6275, 0.4825, 0.5925, 0.835)]),
            ('6.txt', [],
             [(0, 0.5375, 0.3975, 0.735, 0.4325, 0.7, 0.63, 0.5025, 0.595, 0.852),
              (1, 0.5175, 0.395, 0.715, 0.43, 0.68, 0.6275, 0.4825, 0.5925, 0.835)]),
            ('7.txt', [(0, 0.335, 0.11, 0.63, 0.16, 0.5775, 0.4575, 0.2825, 0.405),
                       (1, 0.2275, 0.435, 0.4225, 0.47, 0.3725, 0.765, 0.175, 0.73),
                       (1, 0.425, 0.5725, 0.6225, 0.6075, 0.57, 0.9025, 0.3725, 0.8675)],
             [(0, 0.25, 0.165, 0.53, 0.215, 0.4875, 0.45, 0.21, 0.4025, 0.968),
              (0, 0.5725, 0.2025, 0.77, 0.235, 0.735, 0.4325, 0.5375, 0.3975, 0.567),
              (1, 0.2225, 0.515, 0.5975, 0.5825, 0.545, 0.8775, 0.17, 0.81, 0.685)]),
        ]
        self.num_images = len(data)

        for (txt_name, ground, guess) in data:
            truth_preparer.create(txt_name, ground)
            guess_preparer.create(txt_name, guess)
            self.prepare_image(txt_name.replace('.txt', '.jpg'), ground, guess)

    def prepare_image(self, file_name: str, ground: list[tuple], guess: list[tuple]):
        # Create blank images
        im = Image.new("RGB", IMG_SHAPE, color='black')

        painter = ImageDraw.Draw(im)
        for cxywh in ground:
            color = 'Blue' if cxywh[0] == 0 else 'White'
            # Denormalize the points
            points = [(cxywh[i] * IMG_SHAPE[0], cxywh[i + 1] * IMG_SHAPE[1]) for i in range(1, 9, 2)]
            painter.polygon(points, outline=color, width=6)
        for cxywhf in guess:
            color = 'Yellow' if cxywhf[0] == 0 else 'Green'
            points = [(cxywhf[i] * IMG_SHAPE[0], cxywhf[i + 1] * IMG_SHAPE[1]) for i in range(1, 9, 2)]
            painter.polygon(points, outline=color, width=2)

        im.save(self.img_folder_path / file_name)


@pytest.fixture
def session_setup(box_type: str):
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)

    os.environ['BOX_TYPE'] = box_type
    if box_type == 'rotated':
        fake = FakeRotatedBoxData()
    else:  # 'regular'
        fake = FakeRegularBoxData()

    pytest.yaml_path = fake.yaml_path
    pytest.predict_folder_path = fake.predict_folder_path
