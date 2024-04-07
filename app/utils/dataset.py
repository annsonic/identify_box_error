import contextlib
from copy import deepcopy
import hashlib
from itertools import repeat
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
import re
from typing import Generator, Union

import cv2
import glob
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import yaml

from app.utils import (ROOT, HELP_URL, IMG_FORMATS, DATASET_CACHE_VERSION, NUM_THREADS, TQDM, LOGGER, LOCAL_RANK,
                       is_dir_writeable)
from app.utils.ops import xywh2xyxy


def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.

    Source: ultralytics/utils/__init__.py
    """
    assert Path(file).suffix in (".yaml", ".yml"), f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data


def check_class_names(names):
    """
    Check class names.

    Map imagenet class codes to human-readable names if required. Convert lists to dicts.
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


def check_det_dataset(dataset):
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset descriptor (a YAML file).

    Returns:
        (dict): Parsed dataset information and paths.
    Source: ultralytics/data/utils.py and tailored the code
    """
    # Read YAML
    data = yaml_load(dataset, append_filename=True)  # dictionary
    data["names"] = check_class_names(data["names"])
    path = Path(data.get("yaml_file", "")).parent  # dataset root
    if not path.is_absolute():
        path = (ROOT / path).resolve()

    # Set paths
    data["path"] = path  # download scripts # for validating 'yolo train data=url.zip' usage
    for k in "train", "val", "test":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    return data  # dictionary


def img2label_paths(img_paths):
    """Define label paths as a function of image paths.
    Source: ultralytics/data/utils.py
    """
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path.
    Source: ultralytics/data/dataset.py
    """
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict and convert to python float
    gc.enable()
    return cache


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs).
    Source: ultralytics/data/utils.py
    """
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img: Image.Image):
    """Returns exif-corrected PIL size.
    Source: ultralytics/data/utils.py
    """
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        with contextlib.suppress(Exception):
            exif = img.getexif()
            if exif:
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in [6, 8]:  # rotation 270 or 90
                    s = s[1], s[0]
    return s


def verify_image_label(args):
    """Verify one image-label pair.
    Source: ultralytics/data/utils.py
    """
    im_file, lb_file, num_cls, ndim = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                points = lb[:, 1:]
                assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
                assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                # All labels
                max_cls = lb[:, 0].max()  # max label count
                assert max_cls <= num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    msg = f"WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def save_dataset_cache_file(path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path.
    Source: ultralytics/data/dataset.py
    """
    x["version"] = DATASET_CACHE_VERSION  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
        LOGGER.info(f"New cache created: {path}")
    else:
        LOGGER.warning(f"WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.")


def sanitize_batch(batch, dataset_info):
    """Sanitizes input batch for inference, ensuring correct format and dimensions.
    Source: ultralytics/data/explorer/utils.py
    """
    batch["cls"] = batch["cls"].flatten().astype(int).tolist()
    box_cls_pair = sorted(zip(batch["bboxes"].tolist(), batch["cls"]), key=lambda x: x[1])
    batch["bboxes"] = np.array([np.array(box) for box, _ in box_cls_pair])
    batch["cls"] = [cls for _, cls in box_cls_pair]
    batch["labels"] = [dataset_info["names"][i] for i in batch["cls"]]
    # batch["masks"] = batch["masks"].tolist() if "masks" in batch else [[[]]]
    # batch["keypoints"] = batch["keypoints"].tolist() if "keypoints" in batch else [[[]]]
    return batch


class ExplorerDataset:
    def __init__(self, img_path, data, cache=False):
        self.img_path = img_path
        self.data = data
        self.im_files = self.get_img_files(self.img_path)
        self.label_files = img2label_paths(self.im_files)
        self.labels = self.get_labels()
        self.ni = len(self.labels)  # number of images
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"No images found in {img_path}"
        except Exception as e:
            raise FileNotFoundError(f"Error loading data from {img_path}\n{HELP_URL}") from e

        return im_files

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        Source ultralytics/data/dataset.py and tailored the code
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(len(self.data["names"])),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format="xywh",
                        )
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(path, x)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(
                f"WARNING тЪая╕П No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING тЪая╕П Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(
                f"WARNING тЪая╕П No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.get_image_and_label(index)

    def load_image(self, i: int) -> Union[tuple[np.ndarray, tuple[int, int], tuple[int, int]], tuple[None, None, None]]:
        """Loads 1 image from dataset index 'i' without any resize ops."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]  # orig hw
            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def get_image_and_label(self, index):
        """Get and return label information from the dataset.
        Source: ultralytics/data/dataset.py and tailored the code
        """
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation

        return label


class Explorer:
    def __init__(
            self,
            data: Union[str, Path] = 'coco128.yaml',
            split: str = "train",
    ):
        self.data = check_det_dataset(data)
        self.dataset = self.select_sub_dataset(split)
        self.batch_size = 1

    def select_sub_dataset(self, split: str = "train"):
        if split not in self.data:
            raise ValueError(
                f"Split {split} is not found in the dataset. Available keys in the dataset are {list(self.data.keys())}"
            )

        choice_set = self.data[split]
        choice_set = choice_set if isinstance(choice_set, list) else [choice_set]
        return ExplorerDataset(img_path=choice_set, data=self.data, cache=False)

    def yield_batches(self) -> Generator[dict, None, None]:
        """ The keys of the output batch are:
            'im_file',  # str, absolute path of the image
            'cls',  # list[int], dim(n), n is the number of bounding boxes in the image
            'bboxes',  # ndarray, dim(n, 4), n is the number of bounding boxes in the image
            'segments',  # []
            'keypoints',  # None
            'normalized',  # True
            'bbox_format', # "xywh"
            'img',  # ndarray, dim(h, w, c), h, w, c are the height, width and channel of the image
            'ori_shape', # tuple (height, width)
            'resized_shape', # tuple (height, width)
            'ratio_pad' # tuple (height_ratio=resized/original, width_ratio)
            'labels'  # list[str], dim(n), n is the number of bounding boxes in the image
        """
        for i in tqdm(range(len(self.dataset))):
            batch = self.dataset[i]
            batch = sanitize_batch(batch, self.data)
            batch['bboxes'] = xywh2xyxy(batch['bboxes'])
            batch['bbox_format'] = 'xyxy'

            yield batch
