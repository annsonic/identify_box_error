import numpy as np


from app.utils.metrics import bbox_ioa


class Annotation:
    def __init__(self, color: tuple[int, int, int], name: str, line_width: int = 1):
        self.color = color  # BGR
        self.name = name
        self.line_width = line_width


TRUE_POSITIVE = Annotation((0, 255, 0), "true_positive")  # Green
DUPLICATE = Annotation((29, 170, 255), "duplicate")  # Yellow (Crayola)
WRONG_CLASS = Annotation((0, 121, 255), "wrong_class")  # Safety Orange
BAD_LOCATION = Annotation((255, 255, 0), "bad_location")  # Cyan
WRONG_CLASS_LOCATION = Annotation((220, 209, 255), "wrong_class_location"),  # Baby blue
BACKGROUND = Annotation((255, 128, 0), "background")  # Blue (Crayola)
MISSING = Annotation((255, 0, 0), "missing")  # Red
GROUND_TRUTH = Annotation((0, 0, 0), "ground_truth", line_width=3)  # Black

IOU_FG_TH = 0.4  # IoU > this threshold is a foreground  # TODO: can be set from GUI
IOU_BG_TH = 0.1  # IoU < this threshold is a background


class BoxErrorTypeAnalyzer:
    def __init__(self, gt_data: dict, pd_data: dict, target_class: int):
        self.gt_data = gt_data
        self.pd_data = pd_data
        self.target = target_class
        self.num_gt, self.num_pd = gt_data['bboxes'].shape[0], pd_data['bboxes'].shape[0]
        self.used_gt = [False] * self.num_gt

        self.iou_matrix = bbox_ioa(gt_data['bboxes'], pd_data['bboxes'], iou=True)  # Output dim: (num_gt, num_pd)
        print(f"{self.iou_matrix=}")

    def find_missing_or_true_positive(self):
        # Start from the ground truth side
        for index_gt in range(self.num_gt):
            if self.gt_data['cls'][index_gt] != self.target:
                continue
            current_row_iou = self.iou_matrix[index_gt]
            if sum(current_row_iou < IOU_BG_TH) == self.num_pd:
                self.pd_data['missing_box_errors'].append(index_gt)
                self.used_gt[index_gt] = True
                continue
            candidate_positive_indices = np.where((current_row_iou >= IOU_FG_TH))[0]
            if candidate_positive_indices.size:
                self.used_gt[index_gt] = True
                index_positive = np.argmax(current_row_iou[candidate_positive_indices])
                # In-case of same iou, choose the one with the highest confidence
                index_positive = np.argmax(self.pd_data['conf'][index_positive])
                if self.pd_data['cls'][index_positive] == self.target:
                    self.pd_data['bad_box_errors'][index_positive] = TRUE_POSITIVE
                else:
                    self.pd_data['bad_box_errors'][index_positive] = WRONG_CLASS
                # Check for duplicate boxes
                a = self.pd_data['bboxes'][index_positive].reshape(1, -1)
                for index in candidate_positive_indices:
                    if index == index_positive:
                        continue
                    b = self.pd_data['bboxes'][index].reshape(1, -1)
                    iou = bbox_ioa(a, b, iou=True)
                    if iou > 0.5 and (self.pd_data['cls'][index] == self.target):
                        self.pd_data['bad_box_errors'][index] = DUPLICATE

    def complex_case(self):
        # Initialize the storage
        self.pd_data['bad_box_errors'] = [None] * self.num_pd
        # Iterate from the ground truth side
        self.find_missing_or_true_positive()
        # Iterate from the prediction side
        for index_pd in range(self.num_pd):
            if self.pd_data['bad_box_errors'][index_pd]:
                continue
            current_column_iou = self.iou_matrix[:, index_pd]
            if sum(current_column_iou < IOU_BG_TH) == self.num_gt:
                self.pd_data['bad_box_errors'][index_pd] = BACKGROUND
                continue
            index_gt = np.argmax(current_column_iou)
            if (current_column_iou[index_gt] >= IOU_FG_TH) and (self.pd_data['cls'][index_pd] == self.target):
                self.pd_data['bad_box_errors'][index_pd] = TRUE_POSITIVE
            elif self.pd_data['cls'][index_pd] == self.target:
                self.pd_data['bad_box_errors'][index_pd] = BAD_LOCATION
            else:
                self.pd_data['bad_box_errors'][index_pd] = WRONG_CLASS_LOCATION

    def analyze(self):
        """ Compare the prediction with ground truth and classify the error type.
        The error type is stored in the `error_types` field of the prediction data.
        """
        # Corner cases: no ground truth and no prediction
        if self.num_gt == 0 and self.num_pd == 0:
            self.pd_data['error_types'] = GROUND_TRUTH
            return
        # Corner case: no ground truth
        elif self.num_gt == 0:
            self.pd_data['bad_box_errors'] = [BACKGROUND] * self.num_pd
            return
        # Corner case: no prediction
        elif self.num_pd == 0:
            self.pd_data['missing_box_errors'] = list(range(self.num_gt))
            return

        self.complex_case()
