import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def annotation_to_sensor_obb(
    ann_translation,
    ann_rotation,
    ann_size,
    ego_translation,
    ego_rotation,
    sensor_translation,
    sensor_rotation,
):
    """
    将 nuScenes annotation (世界坐标) 转换为传感器坐标系 OBB
    并返回 o3d.geometry.OrientedBoundingBox

    ann_translation: (3,) 世界坐标中物体中心位置
    ann_rotation: (4,) 世界坐标中物体旋转 [w, x, y, z]
    ann_size: (3,) 物体尺寸 [w, l, h] （完整长度）

    ego_translation: (3,) ego_pose 的位置（世界坐标）
    ego_rotation: (4,) ego_pose 的旋转 [w, x, y, z]（世界→车身）

    sensor_translation: (3,) 传感器相对车身的平移
    sensor_rotation: (4,) 传感器相对车身的旋转 [w, x, y, z]
    """

    # === Step 1: 世界坐标 → 车身坐标 ===
    # 平移
    pos_car = np.asarray(ann_translation) - np.asarray(ego_translation)
    print(pos_car)
    # 旋转（用 ego_pose 的逆旋转）
    rot_world_to_car = R.from_quat(ego_rotation, scalar_first=True).inv()
    pos_car = rot_world_to_car.apply(pos_car)

    print(pos_car)

    # 物体的旋转也要转到车身坐标
    obj_rot_car = rot_world_to_car * R.from_quat(ann_rotation)

    # === Step 2: 车身坐标 → 传感器坐标 ===
    pos_sensor = pos_car - np.asarray(sensor_translation)
    rot_car_to_sensor = R.from_quat(sensor_rotation).inv()
    pos_sensor = rot_car_to_sensor.apply(pos_sensor)

    obj_rot_sensor = rot_car_to_sensor * obj_rot_car

    # Open3D 的 OBB
    obb = o3d.geometry.OrientedBoundingBox(
        center=pos_sensor, R=obj_rot_sensor.as_matrix(), extent=np.asarray(ann_size)
    )
    return obb


"""
annotation
        "translation": [
            2250.4610000000002,
            865.921,
            1.001
        ],
        "size": [
            2.039,
            5.827,
            1.886
        ],
        "rotation": [
            0.7109256040298744,
            0.0,
            0.0,
            -0.7032672219965597
        ],
"""

filename = "n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392"
ego_pose_token = "4f8f06b5d21d4a3c8fda7a075c0d39bf"
"""

  {
 "token": "4f8f06b5d21d4a3c8fda7a075c0d39bf",
 "sample_token": "f9878012c3f6412184c294c13ba4bac3",
 "ego_pose_token": "4f8f06b5d21d4a3c8fda7a075c0d39bf",
 "calibrated_sensor_token": "f577ef2bcba0426a81d4833c5c6febcf",
 "timestamp": 1526915243047392,
 "fileformat": "pcd",
 "is_key_frame": true,
 "height": 0,
 "width": 0,
 "filename": "samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin",
 "prev": "",
 "next": "3edf58dbadcb46729256168be6cf7ddc"
 },

sensor pose:
 "translation": [
 0.891067,
 0.0,
 1.84292
 ],
 "rotation": [
 0.7043825600303035,
 0.002529989518177017,
 -0.0013265948325242168,
 -0.7098147986794471
 ],
 
ego pose
        "rotation": [
            0.9997238940978229,
            8.80899569529068e-05,
            0.006940903896707341,
            0.022448867747433807
        ],
        "translation": [
            2234.259847599768,
            857.4061194452617,
            0.0
        ]
"""

ann_translation = [2250.4610000000002, 865.921, 1.001]
ann_rotation = [0.7109256040298744, 0.0, 0.0, -0.7032672219965597]
ann_size = [2.039, 5.827, 1.886]

ego_translation = [2234.259847599768, 857.4061194452617, 0.0]
ego_rotation = [
    0.9997238940978229,
    8.80899569529068e-05,
    0.006940903896707341,
    0.022448867747433807,
]
sensor_translation = [0.891067, 0.0, 1.84292]
sensor_rotation = [
    0.7043825600303035,
    0.002529989518177017,
    -0.0013265948325242168,
    -0.7098147986794471,
]

bbox = annotation_to_sensor_obb(
    ann_translation,
    ann_rotation,
    ann_size,
    ego_translation,
    ego_rotation,
    sensor_translation,
    sensor_rotation,
)

bbox.color = (0, 1, 0)

geometries = []

def show_pcd(filename: str, geometries: list):
    file_path = "/mnt/data/datasets/nuScenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin"
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    geometries.append(pcd)

geometries.append(bbox)
show_pcd(filename, geometries)

# o3d.visualization.draw_geometries(geometries)
