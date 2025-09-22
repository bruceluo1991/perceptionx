import pytest
import numpy as np

from common.common import BBox2D
from evaluation.evaluation import area_intersection_with_gt, calculate_ap


def test_area_intersection_with_gt():
    detect = BBox2D(x=0, y=0, w=2, h=2)
    gt = BBox2D(x=1, y=1, w=2, h=2)
    intersection_area = area_intersection_with_gt(detect, gt)
    assert isinstance(intersection_area, float)
    assert intersection_area == 1.0


def test_calculate_ap():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    correctness = np.array([True, True, False, True, False])
    num_gt = 4
    ap = calculate_ap(correctness, scores, num_gt)
    assert isinstance(ap, float)
    assert abs(ap - 0.8333) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__])
