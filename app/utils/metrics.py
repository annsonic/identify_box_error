import numpy as np


def bbox_ioa(box1, box2, iou=False, eps=1e-7) -> np.array:
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes are in x1y1x2y2 format.
    Credits: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py

    Args:
        box1 (np.array): A numpy array of shape (n, 4) representing n bounding boxes.
        box2 (np.array): A numpy array of shape (m, 4) representing m bounding boxes.
        iou (bool): Calculate the standard iou if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.array): A numpy array of shape (n, m) representing the intersection over box2 area.
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


class Line:
    """ Represent a line in the form ax + by + c = 0 """
    def __init__(self, v1: np.array, v2: np.array):
        """
        Args:
            v1 (np.array): shape (2,), the first point.
            v2 (np.array): shape (2,), the second point.
        """
        self.a = v2[1] - v1[1]
        self.b = v1[0] - v2[0]
        self.c = v2[0] * v1[1] - v1[0] * v2[1]

    def __repr__(self):
        return f"Line({self.a}x + {self.b}y + {self.c} = 0)"

    def __call__(self, p: np.array) -> float:
        """ Given a point, return whether the point is on the line or not.
        If the result is <=0, the point is on the "inside" of the line.
        If the result is positive, the point is on the "outside" of the line.
        """
        return self.a * p[0] + self.b * p[1] + self.c

    def intersection(self, other) -> np.array:
        """ Given another line, return the intersection point of the two lines.
        # See e.g. https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        """
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a * other.b - self.b * other.a
        return np.array(
            [(self.b * other.c - self.c * other.b)/w,
             (self.c * other.a - self.a * other.c)/w]
        )


def polygon_area(polygon: np.array) -> float:
    """ Calculate the area of a polygon by shoelace formula.
    Args:
        polygon (np.array): shape (4, 2), vertices are ordered in counter-clockwise order.
    Returns:
        float: The area of the polygon.
    """
    return 0.5 * abs(
        np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) -
        np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1))
    )


def polygon_intersection_area(polygon1: np.array, polygon2: np.array) -> float:
    """ Calculate the intersection area of two polygons by geometrical formula.
    Credits: https://stackoverflow.com/a/45268241
    Args:
        polygon1 (np.array): shape (n, 2), vertices are ordered in counter-clockwise order.
        polygon2 (np.array): shape (m, 2), vertices are ordered in counter-clockwise order.
    Returns:
        float: The intersection area of the two polygons.
    """
    # Take the polygon1 as the initial intersection polygon
    intersection = polygon1

    # Loop over the edges of the second polygon
    for p, q in zip(polygon2, np.concatenate((polygon2[1:], polygon2[:1]))):
        if len(intersection) <= 2:
            break  # No intersection

        line = Line(p, q)
        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
                intersection, np.concatenate((intersection[1:], intersection[:1])),
                line_values, np.concatenate((line_values[1:], line_values[:1]))):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0
    return 0.5 * sum(p[0] * q[1] - p[1] * q[0] for p, q in
                     zip(intersection, np.concatenate((intersection[1:], intersection[:1]))))


def obb_iou(segment1: list[np.array], segment2: list[np.array], eps=1e-7) -> np.array:
    """
    Calculate the intersection over union of two sets of oriented bounding boxes.

    Args:
        segment1 (list[np.array]): Length is m, each item is dim(4, 2) representing m oriented bounding boxes.
        segment2 (list[np.array]): Length is n, each item is dim(4, 2) representing n oriented bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
    Returns:
        (np.array): shape (m, n)
    """

    area_1 = np.array([polygon_area(segment) for segment in segment1])
    area_2 = np.array([polygon_area(segment) for segment in segment2])
    intersection = np.array([[
        polygon_intersection_area(segment1[i], segment2[j]) for j in range(len(segment2))]
                             for i in range(len(segment1))])
    print(f'{area_1=}')
    print(f'{area_2=}')
    print(f'{intersection=}')
    return intersection / (area_1 + area_2 + eps - intersection)
