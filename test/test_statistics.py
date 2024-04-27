import pytest

from app.task.study_in_statistics import recorder, Statistician
from conftest import TMP


def get_error_types() -> list[dict]:
    return [
        {'image_file_name': '2.jpg', 'index': 0, 'object_class': 0, 'error_type': 'background'},
        {'image_file_name': '2.jpg', 'index': 1, 'object_class': 0, 'error_type': 'bad_location'},
        {'image_file_name': '2.jpg', 'index': 2, 'object_class': 1, 'error_type': 'true_positive'},
        {'image_file_name': '2.jpg', 'index': 0, 'object_class': 0, 'error_type': 'missing'},
        {'image_file_name': '3.jpg', 'index': 0, 'object_class': 0, 'error_type': 'wrong_class'},
        {'image_file_name': '3.jpg', 'index': 1, 'object_class': 0, 'error_type': 'missing'},
        {'image_file_name': '3.jpg', 'index': 2, 'object_class': 0, 'error_type': 'missing'},
        {'image_file_name': '4.jpg', 'index': 0, 'object_class': 0, 'error_type': 'wrong_class_location'},
        {'image_file_name': '4.jpg', 'index': 1, 'object_class': 0, 'error_type': 'true_positive'},
        {'image_file_name': '4.jpg', 'index': 2, 'object_class': 0, 'error_type': 'duplicate'},
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

    table = Statistician(output_folder_path=TMP).sort_by_errors_per_image(
        recorder.read_csv(TMP / 'test.csv'), target_classes)

    assert table.to_dict(into=dict) == answer[case]
