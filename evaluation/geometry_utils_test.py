import sys

import numpy as np
import pytest


from evaluation.geometry_utils import (
    polygon_area,
    polygon_intersection_with_convex_polygon,
)


def _as_np_float64(points):
    return np.asarray(points, dtype=np.float64)


def _as_tuple_set(poly, decimals=9):
    if poly.size == 0:
        return set()
    return set(tuple(np.round(p, decimals=decimals)) for p in poly)


def test_polygon_area_triangle_orientation():
    tri_ccw = _as_np_float64([[0, 0], [4, 0], [0, 3]])
    tri_cw = tri_ccw[[0, 2, 1]]
    area_ccw = polygon_area(tri_ccw)
    assert isinstance(area_ccw, float)
    area_cw = polygon_area(tri_cw)
    assert isinstance(area_cw, float)
    assert np.isclose(area_ccw, 6.0)
    assert np.isclose(area_cw, 6.0)


def test_polygon_area_rectangle_and_negative_coords():
    rect = _as_np_float64([[-2, -1], [3, -1], [3, 2], [-2, 2]])
    rect_cw = rect[::-1]
    assert np.isclose(polygon_area(rect), 15.0)
    assert np.isclose(polygon_area(rect_cw), 15.0)


def test_polygon_area_degenerate_cases():
    assert np.isclose(polygon_area(_as_np_float64([])), 0.0)
    assert np.isclose(polygon_area(_as_np_float64([[1.0, 2.0]])), 0.0)
    assert np.isclose(polygon_area(_as_np_float64([[0.0, 0.0], [1.0, 0.0]])), 0.0)
    colinear = _as_np_float64([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert np.isclose(polygon_area(colinear), 0.0)


def test_polygon_area_large_values_precision():
    rect = _as_np_float64([[1e6, 0.0], [1e6 + 3.0, 0.0], [1e6 + 3.0, 2.0], [1e6, 2.0]])
    assert np.isclose(polygon_area(rect), 6.0)


def test_intersection_polygon_inside_convex_polygon_returns_polygon():
    poly1 = _as_np_float64(
        [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]
    )  # CCW
    convex2 = _as_np_float64([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])  # CCW
    inter = polygon_intersection_with_convex_polygon(poly1, convex2)
    assert _as_tuple_set(inter) == _as_tuple_set(poly1)
    assert np.isclose(polygon_area(inter), 0.25)


def test_intersection_convex_polygon_inside_polygon_returns_convex():
    poly1 = _as_np_float64(
        [[-10, -10], [10, -10], [10, 10], [-10, 10]]
    )  # CCW large square
    tri_ccw = _as_np_float64([[0, 0], [2, 0], [0, 2]])  # CCW triangle
    inter = polygon_intersection_with_convex_polygon(poly1, tri_ccw)
    assert _as_tuple_set(inter) == _as_tuple_set(tri_ccw)
    assert np.isclose(polygon_area(inter), polygon_area(tri_ccw))


def test_intersection_polygon_cw_or_ccw_same_result():
    # Same polygon1 in CW and CCW should produce identical intersection
    poly1_ccw = _as_np_float64([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])
    poly1_cw = poly1_ccw[::-1]
    convex2 = _as_np_float64(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )  # CCW unit square
    inter_ccw = polygon_intersection_with_convex_polygon(poly1_ccw, convex2)
    inter_cw = polygon_intersection_with_convex_polygon(poly1_cw, convex2)
    assert _as_tuple_set(inter_ccw) == _as_tuple_set(inter_cw)
    assert np.isclose(polygon_area(inter_ccw), 0.36)


def test_intersection_partial_overlap_axis_aligned_squares():
    poly1 = _as_np_float64([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])  # CCW
    convex2 = _as_np_float64([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])  # CCW
    inter = polygon_intersection_with_convex_polygon(poly1, convex2)
    expected = _as_np_float64([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
    assert _as_tuple_set(inter) == _as_tuple_set(expected)
    assert np.isclose(polygon_area(inter), 1.0)


def test_intersection_disjoint_returns_empty():
    poly1 = _as_np_float64([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])  # CCW
    convex2 = _as_np_float64([[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]])  # CCW
    inter = polygon_intersection_with_convex_polygon(poly1, convex2)
    assert isinstance(inter, np.ndarray)
    assert inter.shape == (0, 2)


def test_intersection_touching_along_edge_returns_line_segment_points_area_zero():
    # Shared edge along x=2, y in [0,2]
    poly1 = _as_np_float64([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])  # CCW
    convex2 = _as_np_float64(
        [[2.0, -1.0], [4.0, -1.0], [4.0, 3.0], [2.0, 3.0]]
    )  # CCW vertical strip
    inter = polygon_intersection_with_convex_polygon(poly1, convex2)
    assert np.isclose(polygon_area(inter), 0.0)


def test_intersection_touching_at_single_vertex_returns_point_area_zero():
    # Touch at (1,1) only
    poly1 = _as_np_float64([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])  # CCW
    convex2 = _as_np_float64([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]])  # CCW
    inter = polygon_intersection_with_convex_polygon(poly1, convex2)
    assert _as_tuple_set(inter) == {(1.0, 1.0)}
    assert np.isclose(polygon_area(inter), 0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, '-v']))
