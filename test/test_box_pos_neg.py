from pathlib import Path
import numpy as np
import pytest

from app.task.classify_error_type import (BoxErrorTypeAnalyzer, TRUE_POSITIVE, DUPLICATE, WRONG_CLASS, BAD_LOCATION,
                                          WRONG_CLASS_LOCATION, BACKGROUND,)
from app.utils.metrics import bbox_ioa
from app.utils.parse import truth_prediction_fetcher


@pytest.mark.parametrize("box_type", ['regular'])
def test_iou(session_setup, box_type: str):
    """ Test the calculation of the intersection over union of two boxes """
    fetcher = truth_prediction_fetcher(str(pytest.yaml_path), "test", str(pytest.predict_folder_path))
    next(fetcher)
    fact, guess = next(fetcher)
    iou = bbox_ioa(fact['bboxes'], guess['bboxes'], iou=True)

    assert iou.shape == (fact['bboxes'].shape[0], guess['bboxes'].shape[0])
    assert np.allclose(iou, np.array([[0.033033, 0.0, 0.0], [0.0, 0.11643829, 0.08390914], [0.0, 0.0, 1.0]]))


def get_expected_answer(file_name: str):
    return {
        '1.jpg': {
            'bad_box_errors': [],
            'missing_box_errors': []
        }, 
        '2.jpg': {
            'bad_box_errors': [BACKGROUND, BAD_LOCATION, TRUE_POSITIVE],
            'missing_box_errors': [0]
        }, 
        '3.jpg': {
            'bad_box_errors': [WRONG_CLASS],
            'missing_box_errors': [1, 2]
        }, 
        '4.jpg': {
            'bad_box_errors': [WRONG_CLASS_LOCATION, TRUE_POSITIVE, DUPLICATE],
            'missing_box_errors': []
        }, 
        '5.jpg': {
            'bad_box_errors': [BACKGROUND, DUPLICATE],
            'missing_box_errors': []
        },
        '6.jpg': {
            'bad_box_errors': [BACKGROUND, BACKGROUND],
            'missing_box_errors': []
        },
        '7.jpg': {
            'bad_box_errors': [TRUE_POSITIVE, BACKGROUND, BAD_LOCATION],
            'missing_box_errors': [1]
        },
    }[file_name]


@pytest.mark.parametrize("box_type", ['regular'])
def test_algorithm(session_setup, box_type: str):
    fetcher = truth_prediction_fetcher(str(pytest.yaml_path), "test", str(pytest.predict_folder_path))
    
    for _ in range(7):
        fact, guess = next(fetcher)
        BoxErrorTypeAnalyzer(fact, guess).analyze()

        expected = get_expected_answer(Path(fact['im_file']).name)
        assert len(guess['bad_box_errors']) == len(expected['bad_box_errors'])
        for pd, gd in zip(guess['bad_box_errors'], expected['bad_box_errors']):
            assert pd.name == gd.name
        assert guess['missing_box_errors'] == expected['missing_box_errors']


@pytest.mark.parametrize("box1, box2, answer", [
    (np.array([[0, 100], [0, 0], [100, 0], [100, 100]]), np.array([[50, 0], [0, 0], [0, 50], [50, 50]]), 2500.0),
    (np.array([[50, 100], [50, 0], [150, 0], [150, 100]]), np.array([[50, 0], [0, 0], [0, 50], [50, 50]]), 0.0),
    (np.array([[50, 100], [50, 0], [150, 0], [150, 100]]), np.array([[100, 50], [0, 50], [0, 0], [100, 0]]), 2500.0),
    (np.array([[50, 100], [50, 0], [150, 0], [150, 100]]), np.array([[60, 40], [50, 30], [40, 40], [50, 50]]), 100.0),
])
def test_polygon_intersection_area(box1: np.array, box2: np.array, answer: float):
    from app.utils.ops import sort_vertices_counter_clockwise
    from app.utils.metrics import polygon_intersection_area

    vertices_a = sort_vertices_counter_clockwise(box1)
    vertices_b = sort_vertices_counter_clockwise(box2)
    assert polygon_intersection_area(vertices_a, vertices_b) == answer


@pytest.mark.parametrize("box1, box2, answer", [
    (np.array([[0, 100], [0, 0], [100, 0], [100, 100]]), np.array([[50, 0], [0, 0], [0, 50], [50, 50]]), 0.25),
    (np.array([[50, 100], [50, 0], [150, 0], [150, 100]]), np.array([[50, 0], [0, 0], [0, 50], [50, 50]]), 0.0),
    (np.array([[50, 100], [50, 0], [150, 0], [150, 100]]), np.array([[100, 50], [0, 50], [0, 0], [100, 0]]), 0.2),
    (np.array([[50, 100], [50, 0], [150, 0], [150, 100]]), np.array([[60, 40], [50, 30], [40, 40], [50, 50]]), 0.00990099),
])
def test_obb_iou(box1: np.array, box2: np.array, answer: float):
    from app.utils.ops import sort_vertices_counter_clockwise
    from app.utils.metrics import obb_iou

    vertices_a = sort_vertices_counter_clockwise(box1)
    vertices_b = sort_vertices_counter_clockwise(box2)

    assert obb_iou([vertices_a], [vertices_b])[0][0] == pytest.approx(answer, abs=1e-7)
