import numpy as np


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y


def denormalize_segment(segments: list[np.array], shape: tuple[int, int]) -> np.array:
    """
    Args:
        segments (list[np.array]): A list of numpy arrays representing the segments.
        shape (tuple[int, int]): The shape(height, width) of the image containing the segments.
    Returns:
        list[np.array]: A list of numpy arrays representing the denormalized segments.
    """
    segments = np.array(segments)
    segments[:, :, 0] *= shape[0]
    segments[:, :, 1] *= shape[1]
    return segments


def sort_counter_clockwise_in_numpy_coord(vertices: np.array) -> np.array:
    """
    Args:
        vertices (np.array): shape (4, 2)
    Returns:
        np.array: shape (4, 2), vertices sorted in counter-clockwise order.

    In the numpy coordinate system, the result is in counter-clockwise order
    In the OpenCV coordinate system, the result is in clockwise order.
    Thus, the result matches the DOTA annotation.
    """
    # Calculate the centroid
    x0, y0 = np.mean(vertices, axis=0)
    # Calculate the angle
    radians = np.arctan2(vertices[:, 1] - y0, vertices[:, 0] - x0)
    # Sort the vertices based on the angles in ascending order
    return vertices[np.argsort(radians)]


def xywh2xyxyxyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.zeros((x.shape[0], 8))
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # top right x
    y[..., 3] = x[..., 1] - dh  # top right y
    y[..., 4] = x[..., 0] + dw  # bottom right x
    y[..., 5] = x[..., 1] + dh  # bottom right y
    y[..., 6] = x[..., 0] - dw  # bottom left x
    y[..., 7] = x[..., 1] + dh  # bottom left y
    return y.reshape(-1, 4, 2)
