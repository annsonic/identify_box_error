import csv
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.task.classify_error_type import (TRUE_POSITIVE, DUPLICATE, WRONG_CLASS, BAD_LOCATION,
                                          WRONG_CLASS_LOCATION, BACKGROUND, MISSING)
from app.utils.metrics import average_precision


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
        self.error_type_mapper = {
            'true_positive': TRUE_POSITIVE,
            'duplicate': DUPLICATE,
            'wrong_class': WRONG_CLASS,
            'bad_location': BAD_LOCATION,
            'wrong_class_location': WRONG_CLASS_LOCATION,
            'background': BACKGROUND,
            'missing': MISSING
        }
        self.fp_names = ['background', 'bad_location', 'duplicate', 'wrong_class', 'wrong_class_location']
    
    def pie(self, source: pd.DataFrame, target_classes: list[int] = None) -> pd.Series:
        """ Calculate the proportion of each error type and draw a pie chart.
        The pie chart will be saved in the self.output_folder_path folder.
        Args:
            source (pd.DataFrame): The data to be analyzed.
            target_classes (list[int]): The id of classes to be analyzed.
        Returns:
            pd.Series: The proportion of each error type.
        """
        data = source.copy()

        type_names = list(self.error_type_mapper.keys())
        colors = [x.color for x in self.error_type_mapper.values()]
        colors = [(x[0] / 255.0, x[1] / 255.0, x[2] / 255.0) for x in colors]
        hatches = ['/' if i % 2 == 0 else '' for i in range(len(type_names))]
        file_name = 'error_type_pie.png'
        if target_classes:
            data = data[data['object_class'].isin(target_classes)]
            file_name = f"class({'_'.join(map(str, target_classes))})_{file_name}"

        fractions = data['error_type'].value_counts(normalize=True)

        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(fractions, autopct='%1.1f%%', colors=colors,
                                          wedgeprops=dict(width=0.6), hatch=hatches)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                  bbox=bbox_props, zorder=0, va="center")
        for i, p in enumerate(wedges):
            p.set_edgecolor('grey')
            ang = (p.theta2 - p.theta1) / 2. + p.theta1 + 1e-2  # 1e-2 is the offset for the case(100% proportion)
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))

            horizontal_alignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connection_style = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connection_style})
            ax.annotate(type_names[i], xy=(x, y), xytext=(1.05 * np.sign(x), 1.4 * y),
                        horizontalalignment=horizontal_alignment, **kw)
        ax.set_title('Error Type Distribution')
        plt.savefig(self.output_folder_path / file_name)

        return fractions

    @staticmethod
    def sort_by_errors_per_image(source: pd.DataFrame, target_classes: list[int] = None) -> pd.Series:
        """ Count the number of errors per image and sort the result in descending order.
        Args:
            source (pd.DataFrame): The data to be analyzed.
            target_classes (list[int]): The id of classes to be analyzed.
        Returns:
            pd.Series: The number of errors per image.
        """
        data = source.copy()

        if target_classes:
            data = data[data['object_class'].isin(target_classes)]

        counts = data.groupby('image_file_name')['error_type'].value_counts()
        counts = counts[~counts.index.get_level_values('error_type').isin(['true_positive'])]
        return counts.unstack().fillna(0).sum(axis=1).astype(int)
    
    def histogram(self, source: pd.DataFrame, target_classes: list[int] = None) -> pd.Series:
        """ Draw a histogram of the number of errors per image.
        Args:
            source (pd.DataFrame): The data to be analyzed.
            target_classes (list[int]): The id of classes to be analyzed.
        Returns:
            pd.Series: The number of errors per image.
        """
        data = self.sort_by_errors_per_image(source, target_classes)

        file_name = 'error_type_histogram.png'
        if target_classes:
            file_name = f"class({'_'.join(map(str, target_classes))})_{file_name}"
        if data.empty:
            return data

        plt.hist(x=data.tolist(), bins=range(0, int(data.max()) + 2), histtype='bar', color='skyblue',
                 align='left', rwidth=0.5)
        # Set the tick value type to integer
        plt.xticks(range(0, int(data.max()) + 2))
        plt.yticks(range(0, len(data) + 2))
        plt.xlabel('Number of Errors per Image')
        plt.ylabel('Frequency')
        plt.title('Error Frequency Distribution')
        plt.grid(True)
        plt.savefig(self.output_folder_path / file_name)
        return data
    
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
        for class_id in target_classes:
            # Sorting the data by confidence in descending order, this makes the error with high confidence contributes
            # an important portion in the ap value
            class_data = data[data['object_class'] == class_id].sort_values(['confidence'], ascending=False)
            list_tp = class_data['error_type'].eq('true_positive').tolist()
            list_fn = class_data['error_type'].eq('missing').tolist()
            list_fp = class_data['error_type'].isin(self.fp_names).tolist()
            ap, p, r, p_interp, r_interp = average_precision(list_tp, list_fp, list_fn)
            result[class_id] = ap
            if not pr_curve_name:
                continue

            plt.plot(r, p, color='black')
            if len(p) < 11:  # For the sparse data, annotate the data points
                plt.scatter(r, p, color='black', s=5)
            plt.scatter(r_interp, p_interp, color='red', edgecolor='none', alpha=0.75, s=5)
        
        if pr_curve_name:
            plt.ylim(0, 1.2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('PR Curve')
            plt.legend(['Precision', '101-point interpolated precision'], loc='upper right')
            plt.grid(True)
            plt.savefig(self.output_folder_path / pr_curve_name)
        
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
        if target_classes:
            data = source[source['object_class'].isin(target_classes)]
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

        return maps
        
    def bar_chart_delta_map(self, source: pd.DataFrame, target_classes: list[int] = None, num_classes: int = None
                            ) -> dict[str, float]:

        maps = self.delta_map(source, target_classes, num_classes)

        plt.bar(list(maps.keys()), list(maps.values()), color='grey', hatch='/')
        plt.xticks(rotation=15)
        plt.xlabel('Error Types')
        plt.ylabel('Delta-AP@0.5')
        # Add value annotation above the bar
        for x, y in enumerate(maps.values()):
            plt.text(x, y, f'{y:.4f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(self.output_folder_path / 'delta_map.png')
        return maps
