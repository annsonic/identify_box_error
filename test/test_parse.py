import numpy as np
import pytest

from conftest import TMP, IMG_SHAPE


def test_yaml_load(session_setup):
    from app.utils.parse import yaml_load

    data = yaml_load(str(pytest.yaml_path))

    assert data["names"] == {0: "butterfly", 1: "moth"}
    assert data["yaml_file"] == pytest.yaml_path
    assert data["test"] == TMP / "ground_truth" / "images" / "test"


def partial_expected_cache_content():
    return [
        {
            'im_file': str(TMP / "ground_truth" / "images" / "test" / '1.jpg'),
            'shape': IMG_SHAPE, 'cls': np.array([]), 'bboxes': np.array([]),
            'normalized': True, 'bbox_format': 'xyxy'
        },
        {
            'im_file': str(TMP / "ground_truth" / "images" / "test" / '2.jpg'),
            'shape': IMG_SHAPE, 'cls': np.array([0, 0, 1], dtype=np.int64), 'bboxes': np.array([[0.2, 0.25, 0.4, 0.55], [0.43, 0.38, 0.77, 0.72], [0.4, 0.67, 0.8, 0.93]], dtype=np.float64),
            'normalized': True, 'bbox_format': 'xyxy'
        },
    ]


def test_cache_load(session_setup):
    from app.utils.parse import yaml_load, cache_load

    yaml_data = yaml_load(str(pytest.yaml_path))
    output = cache_load("test", yaml_data)
    expected = partial_expected_cache_content()
    for index in [0, 1]:
        assert output[index]["im_file"] == expected[index]["im_file"]
        assert output[index]["shape"] == expected[index]["shape"]
        assert np.allclose(output[index]["cls"], expected[index]["cls"])
        assert np.allclose(output[index]["bboxes"], expected[index]["bboxes"])
        assert output[index]["normalized"] == expected[index]["normalized"]
        assert output[index]["bbox_format"] == expected[index]["bbox_format"]


def partial_expected_predict_txt_content():
    return {
        '1.txt': {
            'cls': np.array([]),
            'bboxes': np.array([]),
            'normalized': True,
            'bbox_format': 'xyxy',
            'bad_box_errors': None, 
            'missing_box_errors': None},
        '3.txt': {
            'cls': np.array([1]),
            'bboxes': np.array([[0.3, 0.2, 0.5, 0.4]]),
            'normalized': True,
            'bbox_format': 'xyxy',
            'bad_box_errors': None,
            'missing_box_errors': None},
        '4.txt': {
            'cls': np.array([1, 0, 0]),
            'bboxes': np.array([[0.17, 0.48, 0.35, 0.72], [0.6 , 0.3, 0.8, 0.5], [0.58, 0.3 , 0.78, 0.5]]),
            'normalized': True,
            'bbox_format': 'xyxy',
            'bad_box_errors': None, 
            'missing_box_errors': None},
    }


def test_txt_load(session_setup):
    from app.utils.parse import txt_load

    output = txt_load(str(pytest.predict_folder_path))
    expected = partial_expected_predict_txt_content()
    for file in expected.keys():
        assert np.allclose(output[file]["cls"], expected[file]["cls"])
        assert np.allclose(output[file]["bboxes"], expected[file]["bboxes"])
        assert output[file]["normalized"] == expected[file]["normalized"]
        assert output[file]["bbox_format"] == expected[file]["bbox_format"]


def partial_expected_fetcher_output():
    return [
    ({
            'im_file': str(TMP / "ground_truth" / "images" / "test" / '1.jpg'),
            'shape': IMG_SHAPE,
            'cls': np.array([]),
            'bboxes': np.array([]),
            'normalized': True,
            'bbox_format': 'xyxy',
        }, {
            'cls': np.array([]),
            'bboxes': np.array([]),
            'conf': np.array([]),
            'normalized': True,
            'bbox_format': 'xyxy',
            'bad_box_errors': [],
            'missing_box_errors': [],
    }),
    ({
            'im_file': str(TMP / "ground_truth" / "images" / "test" / '2.jpg'),
            'shape': IMG_SHAPE,
            'cls': np.array([0, 0, 1]),
            'bboxes': np.array([[0.2, 0.25, 0.4, 0.55], [0.43, 0.38, 0.77, 0.72], [0.4 , 0.67, 0.8, 0.93]]),
            'normalized': True,
            'bbox_format': 'xyxy',
        }, {
            'cls': np.array([0, 0, 1]),
            'bboxes': np.array([[0.17, 0.48, 0.35, 0.72], [0.6 , 0.1 , 0.8, 0.5], [0.4, 0.67, 0.8, 0.93]]),
            'conf': np.array([0.881, 0.799, 0.697]),
            'normalized': True,
            'bbox_format': 'xyxy',
            'bad_box_errors': [],
            'missing_box_errors': [],
    }),
    ]


def test_fetcher(session_setup):
    from app.utils.parse import truth_prediction_fetcher

    fetcher = truth_prediction_fetcher(str(pytest.yaml_path), "test", str(pytest.predict_folder_path))
    answers = partial_expected_fetcher_output()
    
    for index in range(1):
        fact, guess = next(fetcher)
        expected_fact, expected_guess = answers[index]
        
        assert fact["im_file"] == expected_fact["im_file"]
        assert fact["shape"] == expected_fact["shape"]
        assert np.allclose(fact["cls"], expected_fact["cls"])
        assert np.allclose(fact["bboxes"], expected_fact["bboxes"])
        assert fact["normalized"] == expected_fact["normalized"]
        assert fact["bbox_format"] == expected_fact["bbox_format"]

        assert np.allclose(guess["cls"], expected_guess["cls"])
        assert np.allclose(guess["bboxes"], expected_guess["bboxes"])
        assert np.allclose(guess["conf"], expected_guess["conf"])
        assert guess["normalized"] == expected_guess["normalized"]
        assert guess["bbox_format"] == expected_guess["bbox_format"]
        assert len(guess["bad_box_errors"]) == 0
        assert len(guess["missing_box_errors"]) == 0
