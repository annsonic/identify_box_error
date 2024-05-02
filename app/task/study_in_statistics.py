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
    
    def pie(self, source: pd.DataFrame, target_classes: list[int] = None) -> pd.Series:
        """ Calculate the proportion of each error type and draw a pie chart. """
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
        data = source.copy()

        if target_classes:
            data = data[data['object_class'].isin(target_classes)]

        counts = data.groupby('image_file_name')['error_type'].value_counts()
        counts = counts[~counts.index.get_level_values('error_type').isin(['true_positive'])]
        return counts.unstack().fillna(0).sum(axis=1)
