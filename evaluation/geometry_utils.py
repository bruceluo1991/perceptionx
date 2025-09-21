import numpy as np

from common.common import Point2D, Polygon2D


def polygon_area(polygon: Polygon2D) -> float:
    if len(polygon) < 3:
        return 0.0

    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))).item()


def polygon_intersection_with_convex_polygon(
    polygon1: Polygon2D, convex_polygon2: Polygon2D
) -> Polygon2D:
    """
    Points of convex_polygon2 should be in counter-clockwise order.
    Each edge of convex_polygon2 should have significant length such that the direction of the edge is reliable.
    Points in polygon1 can be clockwise or counter-clockwise.
    """
    # Quick reject if bounding boxes do not overlap
    x_min_1, y_min_1 = np.min(polygon1, axis=0)
    x_max_1, y_max_1 = np.max(polygon1, axis=0)
    x_min_2, y_min_2 = np.min(convex_polygon2, axis=0)
    x_max_2, y_max_2 = np.max(convex_polygon2, axis=0)
    if (x_min_1 > x_max_2 or x_min_2 > x_max_1 or y_min_1 > y_max_2 or y_min_2 > y_max_1):
        return np.zeros((0, 2), dtype=np.float64)

    x_minus_y_1 = polygon1[:, 0] - polygon1[:, 1]
    x_minus_y_2 = convex_polygon2[:, 0] - convex_polygon2[:, 1]
    if (np.max(x_minus_y_1) < np.min(x_minus_y_2) or
        np.max(x_minus_y_2) < np.min(x_minus_y_1)):
        return np.zeros((0, 2), dtype=np.float64)

    x_plus_y_1 = polygon1[:, 0] + polygon1[:, 1]
    x_plus_y_2 = convex_polygon2[:, 0] + convex_polygon2[:, 1]
    if (np.max(x_plus_y_1) < np.min(x_plus_y_2) or
        np.max(x_plus_y_2) < np.min(x_plus_y_1)):
        return np.zeros((0, 2), dtype=np.float64)

    def intersection_with_half_plane(
        polygon: Polygon2D, edge_start: Point2D, edge_end: Point2D
    ) -> Polygon2D:
        """
        The half plane is on the left side of the edge when going from edge_start to edge_end.
        """
        edge_length = np.linalg.norm(edge_end - edge_start)
        if edge_length < 1e-8:
            raise ValueError("Edge length is too small.")
        edge_dir = (edge_end - edge_start) / edge_length  # shape: (2,)
        polygon_translated = polygon - edge_start  # shape: (N, 2)
        cross_prod_2d = (
            polygon_translated[:, 1] * edge_dir[0]
            - polygon_translated[:, 0] * edge_dir[1]
        )  # shape: (N,)
        remaining_points = []
        for i in range(len(polygon)):
            curr_inside = cross_prod_2d[i] >= 0
            prev_inside = cross_prod_2d[i - 1] >= 0
            if curr_inside:
                if not prev_inside:
                    # edge enters the half-plane
                    t = cross_prod_2d[i - 1] / (cross_prod_2d[i - 1] - cross_prod_2d[i])
                    t = np.clip(t, 0.0, 1.0)
                    intersection_point = polygon[i - 1] + t * (
                        polygon[i] - polygon[i - 1]
                    )
                    remaining_points.append(intersection_point)

                remaining_points.append(polygon[i])
            elif prev_inside:
                # edge leaves the half-plane
                t = cross_prod_2d[i - 1] / (cross_prod_2d[i - 1] - cross_prod_2d[i])
                t = np.clip(t, 0.0, 1.0)
                intersection_point = polygon[i - 1] + t * (polygon[i] - polygon[i - 1])
                remaining_points.append(intersection_point)

        if len(remaining_points) == 0:
            return np.zeros((0, 2), dtype=np.float64)

        return np.stack(remaining_points, axis=0)  # shape: (M, 2)

    clipped_polygon = polygon1
    for i in range(len(convex_polygon2)):
        edge_start = convex_polygon2[i - 1]
        edge_end = convex_polygon2[i]
        clipped_polygon = intersection_with_half_plane(
            clipped_polygon, edge_start, edge_end
        )
        if len(clipped_polygon) == 0:
            break

    return clipped_polygon
