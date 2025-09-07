from qs_datasets.nuscenes.pose import Pose
import numpy as np
from typing import List, Optional, Tuple
import os
from scipy.spatial.transform import Rotation
import open3d as o3d

from qs_datasets.nuscenes.constants import DATA_ROOT, QS_METADATA_ROOT

class Frame:
    def __init__(
        self,
        data_root: str,
        main_frame: dict,
        history_frames: List[dict],
        annotation: List[dict],
        calibrated_sensor_pose: Pose,  # 传感器在车坐标系的位姿
    ) -> None:
        """
        文件中的点云坐标为 传感器坐标系
        文件中的标注坐标为 世界坐标系

        标注的translation 的ego坐标系： 前方为x轴，左方为y轴，上方为z轴
        标注的rotation 的ego坐标系： 与ego车头方向一致为0°，绕z轴旋转，顺时针为负

        在输出时，统一转换到qs数据集的车身坐标系: 前方为y轴，右方为x轴，上方为z轴
        """
        self._data_root = data_root
        self._filename: str = main_frame["filename"]
        # history: 0 前一帧，1 前二帧，2 前三帧...
        self._history_ego_poses: List[Pose] = []
        self._history_filenames: List[str] = []
        self._calibrated_sensor_pose: Pose = calibrated_sensor_pose
        self._annotations = annotation

        self._ego_pose = Pose(
            quat=np.array(main_frame["ego_pose"]["rotation"]),
            t=np.array(main_frame["ego_pose"]["translation"]),
        )
        history_poses_sorted = sorted(
            history_frames, key=lambda x: x["timestamp"], reverse=True
        )
        for frame in history_poses_sorted:
            self._history_ego_poses.append(
                Pose(
                    quat=np.array(frame["ego_pose"]["rotation"]),
                    t=np.array(frame["ego_pose"]["translation"]),
                )
            )
            self._history_filenames.append(frame["filename"])

    def _transform_ego_to_qs(self, points: np.ndarray) -> np.ndarray:
        """

        坐标从nuscenes ego转换到qs数据集的车身坐标系
        nuscenes 坐标系: 前方为x轴，左方为y轴，上方为z轴
                   x
                   ^
                   |
                   |
                   |
        y<---------.

        qs数据集坐标系: 前方为y轴，右方为x轴，上方为z轴
                   y
                   ^
                   |
                   |
                   |
                   |
                   .------------>x
        """
        # 转换到qs数据集的车身坐标系
        points[:, [0, 1]] = points[:, [1, 0]]
        points[:, 0] = -points[:, 0]
        return points

    def _load_main(self, main_frame: dict):
        self._filename = main_frame["filename"]
        self._ego_pose = Pose(
            quat=np.array(main_frame["ego_pose"]["rotation"]),
            t=np.array(main_frame["ego_pose"]["translation"]),
        )

    def _load_history(self, history_frames: List[dict]):
        history_poses_sorted = sorted(
            history_frames, key=lambda x: x["timestamp"], reverse=True
        )
        for frame in history_poses_sorted:
            self._history_ego_poses.append(
                Pose(
                    quat=np.array(frame["ego_pose"]["rotation"]),
                    t=np.array(frame["ego_pose"]["translation"]),
                )
            )
            self._history_filenames.append(frame["filename"])

    def _read_pcd_binary(self, filename: str):
        print(os.path.join(self._data_root, filename))
        return np.fromfile(
            os.path.join(self._data_root, filename), dtype=np.float32
        ).reshape(-1, 5)

    def get_points(self, num_history=0):
        """
        读取到的原始点云坐标为 传感器坐标系
        """
        points = self._read_pcd_binary(self._filename)
        # 传感器坐标系转换到车身坐标系
        points[:, :3] = self._calibrated_sensor_pose.transform(points[:, :3])
        # 扩展一列
        points = np.hstack((points, np.zeros((points.shape[0], 1))))
        ids = list(range(len(self._history_filenames)))
        for i in ids[:num_history]:
            history_points = self._read_pcd_binary(self._history_filenames[i])
            history_ego_pose = self._history_ego_poses[i]

            history_points[:, :3] = self._calibrated_sensor_pose.transform(
                history_points[:, :3]
            )

            history_ego_to_current_ego_pose = self._ego_pose.inv().compose(
                history_ego_pose
            )

            history_points[:, :3] = history_ego_to_current_ego_pose.transform(
                history_points[:, :3]
            )
            # 扩展一列
            history_points = np.hstack(
                (
                    history_points,
                    np.ones((history_points.shape[0], 1)) * (i + 1),
                )
            )
            points = np.vstack((points, history_points))
        # 转换到qs数据集的车身坐标系
        points = self._transform_ego_to_qs(points)
        return points

    def get_annotations(self, points: Optional[np.ndarray]=None, min_point_num:int=5) -> np.ndarray:
        """
        output: (N, 8) and num_lidar_pts in the box
        """
        result = []
        num_lidar_pts = []
        for annotation in self._annotations:
            output_annotation = np.zeros(8)  # xyzwlh_yaw_class
            translation_world = np.array(annotation["translation"])
            translation_ego = self._ego_pose.transform_inv(translation_world)
            rotation_world = Rotation.from_quat(
                annotation["rotation"], scalar_first=True
            )
            rotation_ego = self._ego_pose.r.inv() * rotation_world
            output_annotation[:3] = translation_ego
            output_annotation[3:6] = annotation["size"]
            output_annotation[6] = rotation_ego.as_euler("xyz")[2]
            output_annotation[7] = annotation["obstacle_type"]
            num_lidar_pts.append(annotation["num_lidar_pts"])
            result.append(output_annotation)
        output = np.array(result)
        # 转换到qs数据集的车身坐标系
        output[:, :3] = self._transform_ego_to_qs(output[:, :3])
        if points is not None:
            output = self.filter_annotations(output, points, num_lidar_pts, min_point_num)

        return output

    def filter_annotations(
        self,
        annotations: np.ndarray,
        merged_points: np.ndarray,
        num_lidar_pts: List[int],
        min_num_lidar_pts: int,
    ) -> np.ndarray:
        assert len(annotations) == len(num_lidar_pts)
        filtered_annotations = []
        for i in range(len(annotations)):
            if num_lidar_pts[i] > min_num_lidar_pts:
                filtered_annotations.append(annotations[i])
            else:
                # 计算点云在框内的数量
                # 1. 粗过滤
                max_edge = max(annotations[i][3:6])
                center = annotations[i][:3]
                min_x = center[0] - max_edge/2
                max_x = center[0] + max_edge/2
                min_y = center[1] - max_edge/2
                max_y = center[1] + max_edge/2
                min_z = center[2] - max_edge/2
                max_z = center[2] + max_edge/2
                mask = (
                    (merged_points[:, 0] >= min_x)
                    & (merged_points[:, 0] <= max_x)
                    & (merged_points[:, 1] >= min_y)
                    & (merged_points[:, 1] <= max_y)
                    & (merged_points[:, 2] >= min_z)
                    & (merged_points[:, 2] <= max_z)
                )
                filtered_points = merged_points[mask, :3]
                # 2. 精过滤
                num_lidar_pts_in_box = 0
                # 2.1 将点云转换到框的坐标系
                annotation = annotations[i]
                xyz = annotation[:3]
                wlh = annotation[3:6]
                yaw = annotation[6]
                filtered_points = filtered_points - xyz
                # 2.2 旋转点云
                filtered_points = Rotation.from_euler("xyz", [0, 0, -yaw]).apply(filtered_points)
                # 2.3 过滤点云
                filtered_points = filtered_points[
                    (filtered_points[:, 0] >= -wlh[0] / 2)
                    & (filtered_points[:, 0] <= wlh[0] / 2)
                    & (filtered_points[:, 1] >= -wlh[1] / 2)
                    & (filtered_points[:, 1] <= wlh[1] / 2)
                    & (filtered_points[:, 2] >= -wlh[2] / 2)
                    & (filtered_points[:, 2] <= wlh[2] / 2)
                ]
                num_lidar_pts_in_box = len(filtered_points)
                print(num_lidar_pts_in_box, num_lidar_pts[i])
                if num_lidar_pts_in_box >= min_num_lidar_pts:
                    filtered_annotations.append(annotations[i])
        return np.array(filtered_annotations)

if __name__ == "__main__":
    data_root = DATA_ROOT
    from sys import argv
    frame_file = os.path.join(
        QS_METADATA_ROOT,
        f"scene-{argv[1]}",
        "LIDAR_TOP/frames.json"
    )
    import json

    frame_data = json.load(open(frame_file, "r"))
    frames = frame_data["frames"]
    calibrated_sensor_pose = Pose(
        quat=np.array(frame_data["calibrated_sensor"]["rotation"]),
        t=np.array(frame_data["calibrated_sensor"]["translation"]),
    )
    annotation_file = os.path.join(
        QS_METADATA_ROOT,
        f"scene-{argv[1]}",
        "LIDAR_TOP/annotations.json"
    )
    annotation_data = json.load(open(annotation_file, "r"))

    f = frames[0]
    h_fs = frames[1:60]
    f_name = f["filename"]
    frame = Frame(data_root, f, h_fs, annotation_data[f_name], calibrated_sensor_pose)

    points = frame.get_points(2)
    annotations = frame.get_annotations()
    # for annotation in annotations:
    #     if annotation[4] < 10:
    #         continue
    #     print(annotation)

    geometry = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    geometry.append(pcd)
    for annotation in annotations:
        # if annotation[4] < 10:
        #     continue
        box = o3d.geometry.OrientedBoundingBox(
            center=annotation[:3],
            R=Rotation.from_euler("xyz", [0, 0, annotation[6]]).as_matrix(),
            extent=annotation[3:6],
        )
        color_map = {
            1: (1, 0, 0),
            2: (0, 1, 0),
            3: (0, 0, 1),
            4: (1, 1, 0),
        }
        box.color = color_map[annotation[7]]
        geometry.append(box)
    o3d.visualization.draw_geometries(geometry)
