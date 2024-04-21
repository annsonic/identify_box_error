import os
import numpy as np
import pytest

from conftest import TMP, IMG_SHAPE


@pytest.mark.parametrize("box_type", ['regular', 'rotated'])
def test_yaml_load(session_setup, box_type: str):
    from app.utils.parse import yaml_load

    data = yaml_load(str(pytest.yaml_path))

    assert data["names"] == {0: "butterfly", 1: "moth"}
    assert data["yaml_file"] == pytest.yaml_path
    assert data["test"] == TMP / "ground_truth" / "images" / "test"


def partial_expected_cache_content():
    if os.getenv('BOX_TYPE') == 'regular':
        return [
            {
                'im_file': str(TMP / "ground_truth" / "images" / "test" / '1.jpg'),
                'shape': IMG_SHAPE, 'cls': np.array([]), 'bboxes': np.array([]),
                'normalized': True, 'bbox_format': 'xyxy'
            },
            {
                'im_file': str(TMP / "ground_truth" / "images" / "test" / '2.jpg'),
                'shape': IMG_SHAPE, 'cls': np.array([0, 0, 1], dtype=np.int64),
                'bboxes': np.array([[0.2, 0.25, 0.4, 0.55], [0.43, 0.38, 0.77, 0.72], [0.4, 0.67, 0.8, 0.93]],
                                   dtype=np.float64),
                'normalized': True, 'bbox_format': 'xyxy'
            },
        ]
    else:
        return [
            {
                'im_file': str(TMP / "ground_truth" / "images" / "test" / '1.jpg'),
                'shape': IMG_SHAPE, 'cls': np.array([]), 'segments': [],
                'normalized': True,
            },
            {
                'im_file': str(TMP / "ground_truth" / "images" / "test" / '2.jpg'),
                'shape': IMG_SHAPE, 'cls': np.array([0, 0, 1], dtype=np.int64),
                'segments': [np.array([[0.2675, 0.1975], [0.465, 0.2325], [0.4125, 0.5275], [0.2175, 0.4925]]),
                             np.array([[0.47, 0.365], [0.8075, 0.425], [0.7475, 0.76], [0.41, 0.7]]),
                             np.array([[0.39, 0.645], [0.785, 0.7125], [0.7375, 0.9725], [0.345, 0.9025]]),
                             ],
                'normalized': True,
            },
        ]


@pytest.mark.parametrize("box_type", ['regular', 'rotated'])
def test_cache_load(session_setup, box_type: str):
    from app.utils.parse import yaml_load, cache_load

    yaml_data = yaml_load(str(pytest.yaml_path))
    output = cache_load("test", yaml_data, is_obb=(box_type == 'rotated'))
    expected = partial_expected_cache_content()
    for index in [0, 1]:
        assert output[index]["im_file"] == expected[index]["im_file"]
        assert output[index]["shape"] == expected[index]["shape"]
        assert np.allclose(output[index]["cls"], expected[index]["cls"])
        assert output[index]["normalized"] == expected[index]["normalized"]
        if os.getenv('BOX_TYPE') == 'regular':
            assert np.allclose(output[index]["bboxes"], expected[index]["bboxes"])
            assert output[index]["bbox_format"] == expected[index]["bbox_format"]
        else:
            assert np.allclose(output[index]["segments"], expected[index]["segments"])


def partial_expected_predict_txt_content():
    if os.getenv('BOX_TYPE') == 'regular':
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
                'bboxes': np.array([[0.17, 0.48, 0.35, 0.72], [0.6, 0.3, 0.8, 0.5], [0.58, 0.3, 0.78, 0.5]]),
                'normalized': True,
                'bbox_format': 'xyxy',
                'bad_box_errors': None,
                'missing_box_errors': None},
        }
    else:
        return {
            '1.txt': {
                'cls': np.array([]),
                'segments': [],
                'normalized': True,
                'bbox_format': 'xyxy',
                'bad_box_errors': None,
                'missing_box_errors': None},
            '3.txt': {
                'cls': np.array([1]),
                'segments': [np.array([[0.26, 0.2475], [0.4575, 0.2825],
                                      [0.4225, 0.48], [0.225, 0.445]])],
                'normalized': True,
                'bbox_format': 'xyxy',
                'bad_box_errors': None,
                'missing_box_errors': None},
            '4.txt': {
                'cls': np.array([1, 0, 0]),
                'segments': [np.array([[0.0825, 0.5], [0.26, 0.5325], [0.2175, 0.7675], [0.04, 0.7375]]),
                             np.array([[0.5375, 0.3975], [0.735, 0.4325], [0.7, 0.63], [0.5025, 0.595]]),
                             np.array([[0.5175, 0.395], [0.715, 0.43], [0.68, 0.6275], [0.4825, 0.5925]])],
                'normalized': True,
                'bbox_format': 'xyxy',
                'bad_box_errors': None,
                'missing_box_errors': None},
        }


@pytest.mark.parametrize("box_type", ['regular', 'rotated'])
def test_txt_load(session_setup, box_type: str):
    from app.utils.parse import txt_load

    output = txt_load(str(pytest.predict_folder_path), has_conf=True, is_obb=(box_type == 'rotated'))
    expected = partial_expected_predict_txt_content()
    for file in expected.keys():
        assert np.allclose(output[file]["cls"], expected[file]["cls"])
        assert output[file]["normalized"] == expected[file]["normalized"]
        assert output[file]["bbox_format"] == expected[file]["bbox_format"]
        if os.getenv('BOX_TYPE') == 'regular':
            assert np.allclose(output[file]["bboxes"], expected[file]["bboxes"])
        else:
            assert np.allclose(output[file]["segments"], expected[file]["segments"])


def partial_expected_fetcher_output():
    if os.getenv('BOX_TYPE') == 'regular':
        return [(
            {
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
            }), ({
                'im_file': str(TMP / "ground_truth" / "images" / "test" / '2.jpg'),
                'shape': IMG_SHAPE,
                'cls': np.array([0, 0, 1]),
                'bboxes': np.array([[0.2, 0.25, 0.4, 0.55], [0.43, 0.38, 0.77, 0.72], [0.4, 0.67, 0.8, 0.93]]),
                'normalized': True,
                'bbox_format': 'xyxy',
            }, {
                'cls': np.array([0, 0, 1]),
                'bboxes': np.array([[0.13, 0.52, 0.31, 0.76], [0.6, 0.1, 0.8, 0.5], [0.4, 0.67, 0.8, 0.93]]),
                'conf': np.array([0.881, 0.799, 0.697]),
                'normalized': True,
                'bbox_format': 'xyxy',
                'bad_box_errors': [],
                'missing_box_errors': [],
            }),
        ]
    else:
        return [(
            {
                'im_file': str(TMP / "ground_truth" / "images" / "test" / '1.jpg'),
                'shape': IMG_SHAPE,
                'cls': np.array([]),
                'segments': [],
                'normalized': True,
                'bbox_format': 'xyxy',
            }, {
                'cls': np.array([]),
                'segments': [],
                'conf': np.array([]),
                'normalized': True,
                'bbox_format': 'xyxy',
                'bad_box_errors': [],
                'missing_box_errors': [],
            }), (
            {
                'im_file': str(TMP / "ground_truth" / "images" / "test" / '2.jpg'),
                'shape': IMG_SHAPE,
                'cls': np.array([0, 0, 1]),
                'segments': [np.array([[0.2675, 0.1975], [0.465, 0.2325], [0.4125, 0.5275], [0.2175, 0.4925]]),
                             np.array([[0.47, 0.365], [0.8075, 0.425], [0.7475, 0.76], [0.41, 0.7]]),
                             np.array([[0.39, 0.645], [0.785, 0.7125], [0.7375, 0.9725], [0.345, 0.9025]])],
                'normalized': True,
                'bbox_format': 'xyxy',
            }, {
                'cls': np.array([0, 0, 1]),
                'segments': [np.array([[0.1525, 0.4525], [0.33, 0.4825], [0.2875, 0.72], [0.11, 0.6875]]),
                             np.array([[0.6875, 0.12], [0.885, 0.155], [0.815, 0.55], [0.6175, 0.515]]),
                             np.array([[0.39, 0.645], [0.785, 0.7125], [0.7375, 0.9725], [0.345, 0.9025]])],
                'conf': np.array([0.881, 0.799, 0.697]),
                'normalized': True,
                'bbox_format': 'xyxy',
                'bad_box_errors': [],
                'missing_box_errors': [],
            })
        ]


@pytest.mark.parametrize("box_type", ['regular', 'rotated'])
def test_fetcher(session_setup, box_type: str):
    from app.utils.parse import truth_prediction_fetcher

    fetcher = truth_prediction_fetcher(str(pytest.yaml_path), "test", str(pytest.predict_folder_path),
                                       is_obb=(box_type == 'rotated'))
    answers = partial_expected_fetcher_output()
    
    for index in range(2):
        fact, guess = next(fetcher)
        expected_fact, expected_guess = answers[index]

        keys_to_compare = ["normalized", "bbox_format"]
        for key in keys_to_compare:
            assert fact[key] == expected_fact[key]
            assert guess[key] == expected_guess[key]
        key = "bboxes" if os.getenv('BOX_TYPE') == 'regular' else "segments"
        keys_to_compare = ["cls", key]
        for key in keys_to_compare:
            assert np.allclose(fact[key], expected_fact[key])
            assert np.allclose(guess[key], expected_guess[key])
        assert fact["im_file"] == expected_fact["im_file"]
        assert fact["shape"] == expected_fact["shape"]
        assert np.allclose(guess["conf"], expected_guess["conf"])
        assert len(guess["bad_box_errors"]) == 0
        assert len(guess["missing_box_errors"]) == 0
