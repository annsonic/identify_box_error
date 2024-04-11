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
    test_cases:
    1: empty
    2: 1 background, 1 bad location, 1 true positive (iou=0.033, 0.116, 1.0)
    3: 1 wrong class, 2 missing (iou=1.0, 0.0, 0.0)
    4: 1 wrong class location, 1 true positive, 1 duplicate (iou=0.113, 1.0, 0.818)
    5: 1 background, 1 duplicate
    6: 2 backgrounds
    7: 1 true positive, 1 background, 1 bad location (iou=0.444, 0.08, 0.377)
    """
    def __init__(self):
        self.num_images = 0
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
        def rectangle(box: tuple[float]):
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
            painter.rectangle((left,top,right,bottom), outline=color, width=6)
        for cxywhf in guess:
            color = 'Yellow' if cxywhf[0] == 0 else 'Green'
            left, top, right, bottom = rectangle(cxywhf[1:5])
            painter.rectangle((left,top,right,bottom), outline=color, width=2)
        
        im.save(self.img_folder_path / file_name)

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
