from pathlib import Path
import numpy as np
import pytest

from app.task.classify_error_type import (BoxErrorTypeAnalyzer, TRUE_POSITIVE, DUPLICATE, WRONG_CLASS, BAD_LOCATION,
                                          WRONG_CLASS_LOCATION, BACKGROUND, MISSING)
from app.utils.metrics import bbox_ioa
from app.utils.parse import truth_prediction_fetcher


def test_iou(session_setup):
    """ Test the calculation of the intersection over union of two boxes """
    fetcher = truth_prediction_fetcher(str(pytest.yaml_path), "test", str(pytest.predict_folder_path))
    fact, guess = next(fetcher)
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


def test_algorithm(session_setup):
    fetcher = truth_prediction_fetcher(str(pytest.yaml_path), "test", str(pytest.predict_folder_path))
    
    for _ in range(7):
        fact, guess = next(fetcher)
        BoxErrorTypeAnalyzer(fact, guess).analyze()

        expected = get_expected_answer(Path(fact['im_file']).name)
        assert len(guess['bad_box_errors']) == len(expected['bad_box_errors'])
        for pd, gd in zip(guess['bad_box_errors'], expected['bad_box_errors']):
            assert pd.name == gd.name
        assert guess['missing_box_errors'] == expected['missing_box_errors']
