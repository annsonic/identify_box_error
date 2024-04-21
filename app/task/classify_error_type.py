import numpy as np


from app.utils.metrics import bbox_ioa


class Annotation:
    def __init__(self, color: tuple[int, int, int], name: str, line_width: int = 1):
        self.color = color  # BGR
        self.name = name
        self.line_width = line_width

    def __repr__(self):
        return f"<Annotation>name={self.name}, color={self.color}, line_width={self.line_width}"


TRUE_POSITIVE = Annotation((0, 255, 0), "true_positive")  # Green
DUPLICATE = Annotation((29, 170, 255), "duplicate")  # Yellow (Crayola)
WRONG_CLASS = Annotation((0, 121, 255), "wrong_class")  # Safety Orange
BAD_LOCATION = Annotation((255, 255, 0), "bad_location")  # Cyan
WRONG_CLASS_LOCATION = Annotation((220, 209, 255), "wrong_class_location")  # Baby blue
BACKGROUND = Annotation((255, 128, 0), "background")  # Blue (Crayola)
MISSING = Annotation((255, 0, 0), "missing")  # Red
GROUND_TRUTH = Annotation((0, 0, 0), "ground_truth", line_width=3)  # Black

IOU_FG_TH = 0.4  # IoU > this threshold is a foreground  # TODO: can be set from GUI
IOU_BG_TH = 0.1  # IoU < this threshold is a background


class BoxErrorTypeAnalyzer:
    def __init__(self, gt_data: dict, pd_data: dict):
        self.gt_data = gt_data
        self.pd_data = pd_data
        self.num_gt, self.num_pd = gt_data['bboxes'].shape[0], pd_data['bboxes'].shape[0]
        # Initialize the storage
        self.pd_data['bad_box_errors'] = [None] * self.num_pd
        self.matched_gt = set()  # item: index of the ground truth box

        self.iou_matrix = None
        if self.num_gt != 0 and self.num_pd != 0:
            self.iou_matrix = bbox_ioa(gt_data['bboxes'], pd_data['bboxes'], iou=True)  # Output dim: (num_gt, num_pd)

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
            self.matched_gt.add(index_gt)

    def find_duplicate(self):
        indices = np.triu_indices(self.num_pd, k=1)
        duplicates = np.where(self.iou_between_prediction[indices] > 0.5)[0]
        for candidate in duplicates:
            index_1, index_2 = indices[0][candidate].item(), indices[1][candidate].item()
            if self.pd_data['cls'][index_1] != self.pd_data['cls'][index_2]:
                continue
            index_duplicate = index_1 if self.pd_data['conf'][index_2] > self.pd_data['conf'][index_1] else index_2
            self.pd_data['bad_box_errors'][index_duplicate] = DUPLICATE

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
            self.matched_gt.add(index_gt)
        if self.num_gt:
            self.pd_data['missing_box_errors'] = sorted(list(set(range(self.num_gt)) - self.matched_gt))

    def analyze(self):
        """ Compare the prediction with ground truth and classify the error type.
        The error type is stored in the `error_types` field of the prediction data.
        """
        if self.num_pd:
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
