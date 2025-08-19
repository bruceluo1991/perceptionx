import sys
import pytest

import numpy as np

from qs_datasets.nuscenes import pose


def test_pose_transform():
    # Rotation anti-clockwise 90-degree.
    p = pose.Pose(
        np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
        np.array([1, 2, 3]),
    )
    # [1 0 0] -> [0, 1, 0], add [1, 2, 3] -> [1, 3, 3]
    t_p = p.transform(np.array([1, 0, 0]))
    expected = np.array([1., 3., 3.], dtype=np.float32)
    assert np.allclose(t_p, expected)


def test_pose_transform_inv():
    p = pose.Pose(
        np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
        np.array([1, 2, 3]),
    )

    t_p = p.transform_inv(np.array([1, 3, 3]))
    expected = np.array([1., 0., 0.], dtype=np.float32)
    assert np.allclose(t_p, expected)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
