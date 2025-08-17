import json
import open3d as o3d
import numpy as np
import os
from enum import Enum
from scipy.spatial.transform import Rotation

from typing import Tuple

DATA_ROOT = "/mnt/data/datasets/nuScenes"
ANNOTATION_ROOT = "output/generated"

SENSOR_FILE = os.path.join(DATA_ROOT, "v1.0-trainval", "sensor.json")
CALIBRATED_SENSOR_FILE = os.path.join(
    DATA_ROOT, "v1.0-trainval", "calibrated_sensor.json"
)


class TypeEnum(Enum):
    LIDAR_TOP = "LIDAR_TOP"
    ANNOTATION = "ANNOTATION"


def get_sensor_pose_to_ego(sensor_type: TypeEnum) -> Tuple[np.ndarray, np.ndarray]:
    with open(SENSOR_FILE, "r", encoding="utf8") as f:
        sensor_data = json.load(f)
        for sensor in sensor_data:
            if sensor["channel"] == sensor_type.value:
                sensor_token = sensor["token"]
    with open(CALIBRATED_SENSOR_FILE, "r", encoding="utf8") as f:
        calibrated_sensor_data = json.load(f)
        for calibrated_sensor in calibrated_sensor_data:
            if calibrated_sensor["sensor_token"] == sensor_token:
                rotation = np.array(calibrated_sensor["rotation"])
                translation = np.array(calibrated_sensor["translation"])
                return rotation, translation
    return None, None


def get_file_path(filename: str, type: TypeEnum):
    if type == TypeEnum.LIDAR_TOP:
        return os.path.join(DATA_ROOT, "samples", "LIDAR_TOP", f"{filename}.pcd.bin")
    elif type == TypeEnum.ANNOTATION:
        return os.path.join(ANNOTATION_ROOT, "annotation_ego_coord", f"{filename}.json")
    else:
        raise ValueError(f"type not supported: {type}")


def show_pcd(filename: str, geometries: list):
    file_path = get_file_path(filename, TypeEnum.LIDAR_TOP)
    assert os.path.exists(file_path), f"file not exists: {file_path}"
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
    rotation, translation = get_sensor_pose_to_ego(TypeEnum.LIDAR_TOP)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    # geometries.append(pcd)

    translation = np.array([0.891067, 0.0, 1.84292])
    rotation = np.array(
        [
            0.7043825600303035,
            0.002529989518177017,
            -0.0013265948325242168,
            -0.7098147986794471,
        ]
    )

    # print(rotation, translation)
    points[:, :3] = transform_pointcloud_to_ego_coord(
        points[:, :3], rotation, translation
    )
    # # xs = points[:, 0]
    # # ys = points[:, 1]
    # # points[:, 0], points[:, 1] = ys, -xs
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    geometries.append(pcd)


import numpy as np
import open3d as o3d


def generata_bbox(
    position: np.ndarray, theta: float, size: np.ndarray
) -> o3d.geometry.OrientedBoundingBox:
    """
    创建一个绕 z 轴旋转的 OrientedBoundingBox

    position: (3,) 中心坐标 [x, y, z]
    theta: 绕 z 轴旋转角（弧度）
    size: (3,) 框大小 [length_x, length_y, length_z]（完整长度）
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return o3d.geometry.OrientedBoundingBox(center=position, R=R, extent=size)


def transform_pointcloud_to_ego_coord(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """
    将点云从世界坐标转换到ego坐标

    points: (N, 3) 点云坐标
    rotation: (4, 1) 四元数
    translation: (3,) 平移向量
    """
    rotation_matrix = Rotation.from_quat(rotation, scalar_first=True)
    points = rotation_matrix.apply(points) + translation
    return points


def show_annotation(filename: str, geometries: list):
    file_path = get_file_path(filename, TypeEnum.ANNOTATION)
    assert os.path.exists(file_path), f"file not exists: {file_path}"
    with open(file_path, "r", encoding="utf8") as f:
        annotation_data = json.load(f)
    # for annotation in annotation_data:
    #     if annotation["num_lidar_pts"] == 0:
    #         print(annotation)
    #         continue
    #     position = annotation["translation_ego"]
    #     theta = annotation["theta_ego"]
    #     size = annotation["size"]
    #     position = np.array([-position[0], -position[1], position[2]])
    #     position = np.array(position)
    #     theta = np.array(theta)
    #     size = np.array([size[1], size[0], size[2]])
    #     print(position)
    #     bbox = generata_bbox(position, theta, size)
    #     # bbox = generata_bbox(position, theta, np.array([0.3, 0.3, 0.3]))
    #     bbox.color = (0, 1, 0)
    #     geometries.append(bbox)
    # break
    bbox = generata_bbox(
        np.array([16.55157801, 7.77961209, 1.22696056]),
        0.0,
        np.array([2.039, 5.827, 1.886]),
    )
    bbox.color = (1, 0, 0)
    geometries.append(bbox)
    bbox = generata_bbox(
        np.array([-0.7903132373037908, -10.323335484882378, 2.10627348756069]),
        0.0,
        np.array([0.761, 0.689, 1.2496]),
    )
    bbox.color = (1, 0, 0)
    geometries.append(bbox)


filename = "n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392"
geometries = []
show_annotation(filename, geometries)
show_pcd(filename, geometries)
o3d.visualization.draw_geometries(geometries)
# show_pcd(filename)
