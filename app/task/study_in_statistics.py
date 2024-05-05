import csv
from pathlib import Path
from typing import Union

import cv2
import pandas as pd

from app.task.classify_error_type import (TRUE_POSITIVE, DUPLICATE, WRONG_CLASS, BAD_LOCATION,
                                          WRONG_CLASS_LOCATION, BACKGROUND, MISSING)
from app.task.plot import pie_chart, histogram, line_chart, bar_chart, annotate_polygon
from app.utils.metrics import average_precision
from app.utils.ops import xywh2xyxyxyxy, denormalize_segment
from app.utils.parse import load_one_label_file


error_type_mapper = {
    'true_positive': TRUE_POSITIVE,
    'duplicate': DUPLICATE,
    'wrong_class': WRONG_CLASS,
    'bad_location': BAD_LOCATION,
    'wrong_class_location': WRONG_CLASS_LOCATION,
    'background': BACKGROUND,
    'missing': MISSING
}


class Recorder:
    def __init__(self):
        # Note: `object_class` means the ground truth class. The exception is when the error type is `background`.
        self.fieldnames = ['image_file_name', 'index', 'object_class', 'error_type', 'confidence']
    
    def write_csv(self, csv_path: Union[str, Path], analysis_results: list[dict]):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(analysis_results)
    
    def read_csv(self, csv_path: str) -> pd.DataFrame:
        return pd.read_csv(csv_path, header=0, names=self.fieldnames)


recorder = Recorder()


class Statistician:
    def __init__(self, output_folder_path: Path):
        self.output_folder_path = output_folder_path
        self.fp_names = ['background', 'bad_location', 'duplicate', 'wrong_class', 'wrong_class_location']
    
    def error_proportion(self, source: pd.DataFrame, target_classes: list[int] = None) -> pd.Series:
        """ Calculate the proportion of each error type and draw a pie chart.
        The pie chart will be saved in the self.output_folder_path folder.
        Args:
            source (pd.DataFrame): The data to be analyzed.
            target_classes (list[int]): The id of classes to be analyzed.
        Returns:
            pd.Series: The proportion of each error type.
        """
        data = source.copy()

        type_names = list(error_type_mapper.keys())
        colors = [x.color for x in error_type_mapper.values()]
        colors = [(x[0] / 255.0, x[1] / 255.0, x[2] / 255.0) for x in colors]
        hatches = ['/' if i % 2 == 0 else '' for i in range(len(type_names))]
        file_name = 'error_type_pie.png'
        if target_classes:
            data = data[data['object_class'].isin(target_classes)]
            file_name = f"class({'_'.join(map(str, target_classes))})_{file_name}"

        fractions = data['error_type'].value_counts(normalize=True)
        pie_chart(fractions, colors, hatches, type_names, self.output_folder_path / file_name)
        return fractions

    def sort_by_errors_per_image(self, source: pd.DataFrame, target_classes: list[int] = None) -> pd.Series:
        """ Count the number of errors per image and sort the result in descending order.
        Args:
            source (pd.DataFrame): The data to be analyzed.
            target_classes (list[int]): The id of classes to be analyzed.
        Returns:
            pd.Series: The number of errors per image.
        """
        data = source.copy()
        file_name = 'error_type_histogram.png'
        if target_classes:
            data = data[data['object_class'].isin(target_classes)]
            file_name = f"class({'_'.join(map(str, target_classes))})_{file_name}"

        counts = data.groupby('image_file_name')['error_type'].value_counts()
        counts = counts[~counts.index.get_level_values('error_type').isin(['true_positive'])]
        result = counts.unstack().fillna(0).sum(axis=1).astype(int)

        data = result.tolist()
        bins = max(data) if data else 10
        histogram(data, bins, self.output_folder_path / file_name)
        return result
    
    def map(self, data: pd.DataFrame, target_classes: list[int], pr_curve_name: str = None) -> dict[int, float]:
        """ Calculate average precision of each class. Optionally, draw the PR curve.
        * Average Precision (AP): area under the PR curve

        Args:
            data (DataFrame): the data to be analyzed.
            target_classes (list[int]): the id of classes to be analyzed.
            pr_curve_name (str): the name of the PR curve image file.
                                 The file will be saved in the self.output_folder_path folder.
        Returns:
            result (dict[int, float]): the AP for each class.
        """
        if data.empty:
            return {}
        result = dict()
        intermediate_result = dict()
        for class_id in target_classes:
            # Sorting the data by confidence in descending order, this makes the error with high confidence contributes
            # an important portion in the ap value
            class_data = data[data['object_class'] == class_id].sort_values(['confidence'], ascending=False)
            list_tp = class_data['error_type'].eq('true_positive').tolist()
            list_fn = class_data['error_type'].eq('missing').tolist()
            list_fp = class_data['error_type'].isin(self.fp_names).tolist()
            ap, p, r, p_interp, r_interp = average_precision(list_tp, list_fp, list_fn)
            result[class_id] = ap
            if pr_curve_name:
                intermediate_result[class_id] = (r, p, r_interp, p_interp)

        if pr_curve_name:
            line_chart(intermediate_result, self.output_folder_path / pr_curve_name)
        
        return result
    
    def delta_map(self, source: pd.DataFrame, target_classes: list[int] = None, num_classes: int = None
                  ) -> dict[str, float]:
        """ What mAP would have been if there is no this kind of error. 
        * delta_map is defined as the (metric_after_fixing - metric_before_fixing).
        * Note that all error impacts and the metric will not add to 1.

        Args:
            source (pd.DataFrame): The data to be analyzed.
            target_classes (list[int]): The id of classes to be analyzed.
            num_classes (int): The total number of classes in the original dataset.
                               We need `num_classes` when the `target_classes` is None.
        Returns:
            dict[str, float]: The delta-AP for each error type.
        """
        file_name = 'delta_map.png'
        if target_classes:
            data = source[source['object_class'].isin(target_classes)]
            file_name = f"class({'_'.join(map(str, target_classes))})_{file_name}"
        else:
            data = source.copy()
            target_classes = range(num_classes)

        maps = dict()
        for error_type in ([None, 'missing'] + self.fp_names):
            aps = self.map(data[data['error_type'] != error_type], target_classes)
            if not error_type:
                error_type = 'baseline'
            if not aps:
                maps[error_type] = 0
            else:
                maps[error_type] = sum(aps.values()) / len(target_classes)
        baseline = maps.pop('baseline')
        for error_name, m_ap in maps.items():
            maps[error_name] = m_ap - baseline
        bar_chart(maps, self.output_folder_path / file_name)

        return maps


class PolygonAnnotator:
    def __init__(self, src_img_folder: str, label_folder: str, prediction_folder: str, dst_img_folder: str,
                 analyzed: pd.DataFrame):
        self.src_img_folder = Path(src_img_folder)
        self.label_folder = Path(label_folder)
        self.prediction_folder = Path(prediction_folder)
        self.dst_img_folder = Path(dst_img_folder)
        self.dst_img_folder.mkdir(parents=True, exist_ok=True)
        self.analyzed = analyzed

    def run(self):
        grouped = self.analyzed.groupby('image_file_name')

        for image_file_name, df in grouped:
            txt_file_name = Path(str(image_file_name)).stem + '.txt'
            gt_classes, gt_boxes = load_one_label_file(str(self.label_folder / txt_file_name), has_conf=False)
            pd_classes, pd_boxes, _ = load_one_label_file(str(self.prediction_folder / txt_file_name), has_conf=True)
            img = cv2.imread(str(self.src_img_folder / str(image_file_name)))
            size = (img.shape[0], img.shape[1])
            if gt_boxes.shape[1] == 4:
                gt_boxes = xywh2xyxyxyxy(gt_boxes)
            elif gt_boxes.shape[1] == 8:
                gt_boxes = gt_boxes.reshape(-1, 4, 2)
            if pd_boxes.shape[1] == 4:
                pd_boxes = xywh2xyxyxyxy(pd_boxes)
            elif pd_boxes.shape[1] == 8:
                pd_boxes = pd_boxes.reshape(-1, 4, 2)
            if gt_boxes.size:
                gt_boxes = denormalize_segment(gt_boxes, size)
            if pd_boxes.size:
                pd_boxes = denormalize_segment(pd_boxes, size)
            annotations = []
            for _, row in df.iterrows():
                if row['error_type'] == 'missing':  # Read the ground truth label
                    polygon = gt_boxes[row['index']]
                    color = error_type_mapper['missing'].color
                else:  # Read the prediction label
                    polygon = pd_boxes[row['index']]
                    color = error_type_mapper[row['error_type']].color
                annotations.append((polygon, color))
            annotate_polygon(img, annotations, str(self.dst_img_folder / image_file_name))
