import numpy as np
import pytest

from app.task.study_in_statistics import recorder, Statistician
from app.utils.metrics import average_precision
from conftest import TMP


def get_error_types() -> list[dict]:
    return [
        {'image_file_name': '2.jpg', 'index': 0, 'object_class': 0, 'error_type': 'background', 'confidence': 0.3},
        {'image_file_name': '2.jpg', 'index': 1, 'object_class': 0, 'error_type': 'bad_location', 'confidence': 0.7},
        {'image_file_name': '2.jpg', 'index': 2, 'object_class': 1, 'error_type': 'true_positive', 'confidence': 0.9},
        {'image_file_name': '2.jpg', 'index': 0, 'object_class': 0, 'error_type': 'missing', 'confidence': 0},
        {'image_file_name': '3.jpg', 'index': 0, 'object_class': 0, 'error_type': 'wrong_class', 'confidence': 0.9},
        {'image_file_name': '3.jpg', 'index': 1, 'object_class': 0, 'error_type': 'missing', 'confidence': 0},
        {'image_file_name': '3.jpg', 'index': 2, 'object_class': 0, 'error_type': 'missing', 'confidence': 0},
        {'image_file_name': '4.jpg', 'index': 0, 'object_class': 0, 'error_type': 'wrong_class_location',
         'confidence': 0.35},
        {'image_file_name': '4.jpg', 'index': 1, 'object_class': 0, 'error_type': 'true_positive', 'confidence': 0.9},
        {'image_file_name': '4.jpg', 'index': 2, 'object_class': 0, 'error_type': 'duplicate', 'confidence': 0.6},
    ]


def test_write_read_csv():
    answers = get_error_types()

    recorder.write_csv(TMP / 'test.csv', answers)
    outputs = recorder.read_csv(TMP / 'test.csv')

    assert outputs.to_dict('records') == answers


def get_expected_proportions(case: int) -> dict[str, float]:
    return {
        0: {
            'missing': 0.3,
            'true_positive': 0.2,
            'background': 0.1,
            'bad_location': 0.1,
            'wrong_class': 0.1,
            'wrong_class_location': 0.1,
            'duplicate': 0.1
        },
        1: {
            'missing': 0.3,
            'true_positive': 0.2,
            'background': 0.1,
            'bad_location': 0.1,
            'wrong_class': 0.1,
            'wrong_class_location': 0.1,
            'duplicate': 0.1
        },
        2: {'true_positive': 1.0},
    }[case]


@pytest.mark.parametrize("target_classes, case", [
    (None, 0),  # all data
    ([1, 0], 1),  # target_classes
    ([1], 2)  # target_classes
])
def test_pie_chart(target_classes: list[int], case: int):
    data = get_error_types()
    recorder.write_csv(TMP / 'test.csv', data)

    proportions = Statistician(output_folder_path=TMP).pie(
        recorder.read_csv(TMP / 'test.csv'), target_classes)

    assert proportions.to_dict(into=dict) == get_expected_proportions(case)


@pytest.mark.parametrize("target_classes, case", [
    (None, 0),  # all data
    ([1, 0], 1),  # target_classes
    ([1], 2)  # target_classes
])
def test_sort(target_classes: list[int], case: int):
    data = get_error_types()
    recorder.write_csv(TMP / 'test.csv', data)
    answer = {
        0: {'2.jpg': 3.0, '3.jpg': 3.0, '4.jpg': 2.0},
        1: {'2.jpg': 3.0, '3.jpg': 3.0, '4.jpg': 2.0},
        2: {}  # has true_positive, no errors
    }

    table = Statistician(output_folder_path=TMP).histogram(
        recorder.read_csv(TMP / 'test.csv'), target_classes)

    assert table.to_dict(into=dict) == answer[case]


@pytest.mark.parametrize("case", [
    {
        "list_tp": [],
        "list_fp": [],
        "list_fn": [],
        "average_precision": 0.0,
        "precision": np.array([]),
        "recall": np.array([]),
        "smoothed_precision": np.array([0.0] * 101)
    }, {
        "list_tp": [True, True, False, False, True, False],
        "list_fp": [False, False, True, True, False, False],
        "list_fn": [False, False, False, False, False, True],
        "average_precision": 0.643564356,
        "precision": np.array([1.0, 1.0, 2/3.0, 2/4.0, 3/5.0, 3/5.0]),
        "recall": np.array([1/4.0, 2/4.0, 2/4.0, 2/4.0, 3/4.0, 3/4.0]),
        "smoothed_precision": np.array([1.0] * 50 + [0.6] * 25 + [0] * 26)
    },
    {  # Ref[2]
        "list_tp": [True, False, True, False, False, True, False],
        "list_fp": [False, True, False, True, True, False, True],
        "list_fn": [False] * 7,
        "average_precision": 0.717821782,
        "precision": np.array([1.0, 0.5, 2/3.0, 0.5, 0.4, 0.5, 3/7.0]),
        "recall": np.array([1/3.0, 1/3.0, 2/3.0, 2/3.0, 2/3.0, 1.0, 1.0]),
        "smoothed_precision": np.array([1.0] * 34 + [2/3.0] * 33 + [0.5] * 33 + [0])
    }])
def test_average_precision(case: dict):
    """
    # Ref[2] https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
    The answer AP in Ref[2] is 0.727272727 in 11-point.
    However, in the 101-point `interpolated` AP definition, the answer is 0.717821782.
    """
    ap, p, r, smooth_p, r_axis = average_precision(case['list_tp'], case['list_fp'], case['list_fn'])

    assert len(r_axis) == 101
    # Compare the floating point number
    assert abs(ap - case['average_precision']) < 1e-4
    assert np.allclose(p, case['precision'], atol=1e-4)
    assert np.allclose(r, case['recall'], atol=1e-4)
    assert np.allclose(smooth_p, case['smoothed_precision'], atol=1e-4)


@pytest.mark.parametrize("target_classes, num_classes, expected", [
    (None, 2, {'missing': 0.18564356, 'background': 0.0, 'bad_location': 0.0, 'duplicate': 0.0,
               'wrong_class': 0.06188119, 'wrong_class_location': 0.0}),
    ([0, 1], 2, {'missing': 0.18564356, 'background': 0.0, 'bad_location': 0.0, 'duplicate': 0.0,
                 'wrong_class': 0.06188119, 'wrong_class_location': 0.0}),
    ([0], 1, {'missing': 0.37128713, 'background': 0.0, 'bad_location': 0.0, 'duplicate': 0.0,
              'wrong_class': 0.12376238, 'wrong_class_location': 0.0})
])
def test_delta_mean_average_precision(target_classes: list[int], num_classes: int, expected: dict[str, float]):
    data = get_error_types()
    recorder.write_csv(TMP / 'test.csv', data)
    output = Statistician(output_folder_path=TMP).delta_map(
        recorder.read_csv(TMP / 'test.csv'), target_classes=target_classes, num_classes=num_classes)

    for key, value in expected.items():
        assert abs(output[key] - value) < 1e-4


def test_bar():
    data = get_error_types()
    recorder.write_csv(TMP / 'test.csv', data)
    output = Statistician(output_folder_path=TMP).bar_chart_delta_map(
        recorder.read_csv(TMP / 'test.csv'), target_classes=None, num_classes=2)
