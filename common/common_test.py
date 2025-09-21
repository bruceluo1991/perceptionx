import pytest

from common.common import BBox2D, BBox3D


def test_bbox2d_area():
    a = BBox2D([0.0, 0.0, 2.0, 5.0, 0.0])
    area = a.area()
    assert area == 10.0
    assert isinstance(area, float)


def test_bbox3d():
    b = BBox3D([0.0, 0.0, 0.0, 2.0, 5.0, 3.0, 0.0])
    volume = b.volume()
    assert volume == 30.0
    assert isinstance(volume, float)
    assert isinstance(b.z, float)
    assert isinstance(b.h, float)
    assert isinstance(b.bbox2d(), BBox2D)

if __name__ == "__main__":
    pytest.main([__file__])
