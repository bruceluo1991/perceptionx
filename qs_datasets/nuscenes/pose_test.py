import numpy as np

from qs_datasets.nuscenes import pose


def _test():
    # Rotation anti-clockwise 90-degree.
    p = pose.Pose(
        np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]), np.array([1, 2, 3]),)
    # [1 0 0] -> [0, 1, 0], add [1, 2, 3] -> [1, 3, 3]
    t_p = p.transform(np.array([1, 0, 0]))
    assert np.all(t_p == [1, 3, 3])
    print(p, t_p)

    print(p.transform_inv(np.array([1, 0, 0])))


if __name__ == "__main__":
    _test()
