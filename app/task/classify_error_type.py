import numpy as np
from pathlib import Path

from app.utils.metrics import bbox_ioa, obb_iou


class Annotation:
    def __init__(self, color: tuple[int, int, int], name: str, line_width: int = 1):
        self.color = color  # BGR
        self.name = name
        self.line_width = line_width

    def __repr__(self):
        return f"<Annotation>name={self.name}, color={self.color}, line_width={self.line_width}"


# Color code: RGB
TRUE_POSITIVE = Annotation((229, 228, 226), "true_positive")  # Platinum
DUPLICATE = Annotation((247, 127, 190), "duplicate")  # Persian Pink
WRONG_CLASS = Annotation((137, 207, 240), "wrong_class")  # Baby blue
BAD_LOCATION = Annotation((195, 176, 145), "bad_location")  # Khaki
WRONG_CLASS_LOCATION = Annotation((144, 238, 144), "wrong_class_location")  # Light green
BACKGROUND = Annotation((181, 126, 220), "background")  # Lavender
MISSING = Annotation((240, 128, 128), "missing")  # Light coral
GROUND_TRUTH = Annotation((0, 0, 0), "ground_truth", line_width=3)  # Black

IOU_FG_TH = 0.4  # IoU > this threshold is a foreground  # TODO: can be set from GUI
IOU_BG_TH = 0.1  # IoU < this threshold is a background


class BoxErrorTypeAnalyzer:
    """ Given a ground truth file and a prediction file, analyze the error type of each predicted box."""
    def __init__(self, gt_data: dict, pd_data: dict, is_obb: bool = False):

        self.gt_data, self.pd_data = gt_data, pd_data
        if is_obb:
            self.num_gt = len(gt_data['segments'])
            self.num_pd = len(pd_data['segments']) if pd_data else 0
        else:
            self.num_gt = gt_data['bboxes'].shape[0]
            self.num_pd = pd_data['bboxes'].shape[0] if pd_data else 0
        # Initialize the storage
        self.pd_data['bad_box_errors'] = [None] * self.num_pd
        self.matched_gt = set()  # item: index of the ground truth box

        self.iou_matrix = None  # dim: (num_gt, num_pd)
        if self.num_gt != 0 and self.num_pd != 0:
            if is_obb:
                self.iou_matrix = obb_iou(gt_data['segments'], pd_data['segments'])
            else:
                self.iou_matrix = bbox_ioa(gt_data['bboxes'], pd_data['bboxes'], iou=True)
        # self.iou_between_prediction dim: (num_pd, num_pd)
        if self.num_pd != 0:
            if is_obb:
                self.iou_between_prediction = obb_iou(pd_data['segments'], pd_data['segments'])
            else:
                self.iou_between_prediction = bbox_ioa(self.pd_data['bboxes'], self.pd_data['bboxes'], iou=True)

    def true_positive_or_wrong_class(self, index_gt: int):
        current_row_iou = self.iou_matrix[index_gt]
        candidate_positive_indices = np.where((current_row_iou >= IOU_FG_TH))[0]
        if candidate_positive_indices.size:
            index_candidate = np.flatnonzero(current_row_iou == np.amax(current_row_iou))
            # In-case of same iou, choose the one with the highest confidence
            if index_candidate.size > 1:
                index_candidate = index_candidate[np.argmax(self.pd_data['conf'][index_candidate])]
            else:
                index_candidate = index_candidate.item()
            # True positive is not an error, but we need to mark the box has been examined
            if self.pd_data['cls'][index_candidate] == self.gt_data['cls'][index_gt]:
                self.pd_data['bad_box_errors'][index_candidate] = TRUE_POSITIVE
            else:
                self.pd_data['bad_box_errors'][index_candidate] = WRONG_CLASS
                self.pd_data['corrected_cls'][index_candidate] = self.gt_data['cls'][index_gt]
            self.matched_gt.add(index_gt)

    def find_duplicate(self):
        # Separate the boxes into class groups
        class_box_mapper = {cls: [] for cls in set(self.pd_data['cls'])}  # key: class, value: list of box indices
        for index, cls in enumerate(self.pd_data['cls']):
            class_box_mapper[cls].append(index)
        # Find the boxes which have IoU > 0.5 with each other
        masks = np.triu_indices(self.num_pd, k=1)  # Exclude the diagonal of the matrix
        candidates = np.where(self.iou_between_prediction[masks] > 0.5)[0]  # value is the index of the flattened mask
        # Compare the confidence of the grouped boxes
        for cls in class_box_mapper:
            if len(class_box_mapper[cls]) < 2:
                continue
            scores = [self.pd_data['conf'][index] for index in class_box_mapper[cls]]
            index_main = class_box_mapper[cls][np.argmax(scores)]
            for candidate in candidates:
                index_1, index_2 = masks[0][candidate].item(), masks[1][candidate].item()
                if index_1 == index_main:
                    self.pd_data['bad_box_errors'][index_2] = DUPLICATE
                if index_2 == index_main:
                    self.pd_data['bad_box_errors'][index_1] = DUPLICATE

    def complex_case(self):
        # Iterate from the ground truth side
        for index_gt in range(self.num_gt):
            self.true_positive_or_wrong_class(index_gt)
        # Iterate from the prediction side
        for index_pd in range(self.num_pd):
            if self.pd_data['bad_box_errors'][index_pd]:
                continue
            current_column_iou = self.iou_matrix[:, index_pd]

            if sum(current_column_iou < IOU_BG_TH) == self.num_gt:
                self.pd_data['bad_box_errors'][index_pd] = BACKGROUND
                continue

            index_gt = np.argmax(current_column_iou)
            if self.pd_data['cls'][index_pd] == self.gt_data['cls'][index_gt]:
                self.pd_data['bad_box_errors'][index_pd] = BAD_LOCATION
            else:
                self.pd_data['bad_box_errors'][index_pd] = WRONG_CLASS_LOCATION
            self.pd_data['corrected_cls'][index_pd] = self.gt_data['cls'][index_gt]
            self.matched_gt.add(index_gt)
        if self.num_gt:
            self.pd_data['missing_box_errors'] = sorted(list(set(range(self.num_gt)) - self.matched_gt))

    def parse_analysis_results(self) -> list[dict]:
        """ Format the results for the class Recorder to write to the csv file """
        results = []
        if self.num_pd == 0:
            return results
        for index, cls in enumerate(self.pd_data['cls']):
            type_name = self.pd_data['bad_box_errors'][index].name
            results.append({
                'image_file_name': Path(self.gt_data['im_file']).name,
                'index': index,
                'object_class': self.pd_data['corrected_cls'][index],
                'error_type': type_name,
                'confidence': self.pd_data['conf'][index]
            })
        for index in self.pd_data['missing_box_errors']:
            results.append({
                'image_file_name': Path(self.gt_data['im_file']).name,
                'index': index,
                'object_class': self.gt_data['cls'][index],
                'error_type': MISSING.name,
                'confidence': 0
            })
        return results

    def analyze(self):
        """ Compare the prediction with ground truth and classify the error type.
        The error type is stored in the `error_types` field of the prediction data.
        """
        # We need to find the duplicates first, in-case num_gt = 0 and exits without checking the predicted boxes
        if self.num_pd > 1:
            self.find_duplicate()
        # Corner cases: no ground truth and no prediction
        if self.num_gt == 0 and self.num_pd == 0:
            return
        # Corner case: no ground truth
        elif self.num_gt == 0:
            for index in range(self.num_pd):
                if not self.pd_data['bad_box_errors'][index]:
                    self.pd_data['bad_box_errors'][index] = BACKGROUND
            return
        # Corner case: no prediction
        elif self.num_pd == 0:
            self.pd_data['missing_box_errors'] = list(range(self.num_gt))
            return

        self.complex_case()
