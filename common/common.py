from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


Point2D = NDArray[np.float64]  # shape: (2,)
Polygon2D = NDArray[np.float64]  # shape: (N, 2)


# When yaw = 0, the length is along the y-axis.
class BBox2D(np.ndarray):
    def __new__(cls, input_array) -> "BBox2D":
        # (x_center, y_center, width, length, yaw)
        obj = np.array(input_array, dtype=np.float64).view(cls)
        if obj.shape != (5,):
            raise ValueError("Input array must have shape (5,)")
        return obj

    def area(self) -> float:
        _, _, width, length, _ = self
        return width.item() * length.item()

    def corners(self) -> Polygon2D:
        x_center, y_center, width, length, yaw = self
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        half_width = width / 2
        half_length = length / 2

        corners = np.array(
            [
                [-half_width, -half_length],
                [half_width, -half_length],
                [half_width, half_length],
                [-half_width, half_length],
            ]
        )  # shape: (4, 2)

        rotation_matrix = np.array(
            [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]]
        )  # shape: (2, 2)

        rotated_corners = corners @ rotation_matrix.T  # shape: (4, 2)
        translated_corners = rotated_corners + np.array(
            [x_center, y_center]
        )  # shape: (4, 2)

        return translated_corners  # shape: (4, 2)


class BBox3D(NDArray[np.float64]):
    def __new__(cls, input_array) -> "BBox3D":
        # (x_center, y_center, z_center, width, length, height, yaw)
        obj = np.array(input_array, dtype=np.float64).view(cls)
        if obj.shape != (7,):
            raise ValueError("Input array must have shape (7,)")
        return obj

    def volume(self) -> float:
        _, _, _, width, length, height, _ = self
        return width.item() * length.item() * height.item()

    def bbox2d(self) -> BBox2D:
        x_center, y_center, _, width, length, _, yaw = self
        return BBox2D([x_center, y_center, width, length, yaw])

    @property
    def z(self) -> float:
        return self[2].item()

    @property
    def h(self) -> float:
        return self[5].item()


class DetectionObject(NamedTuple):
    bbox3d: BBox3D
    type_id: int
    score: float

class LabelObject(NamedTuple):
    bbox3d: BBox3D
    type_id: int
