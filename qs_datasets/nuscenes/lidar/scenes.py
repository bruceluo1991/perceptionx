import json
import os
from qs_datasets.nuscenes.pose import Pose
from qs_datasets.nuscenes.lidar.frame import Frame
from qs_datasets.nuscenes.constants import QS_METADATA_ROOT

import numpy as np

from typing import Optional, List


class Scene:
    def __init__(self, raw_data_folder: str, scene_folder: str) -> None:
        self._raw_data_folder = raw_data_folder
        self._scene_folder = scene_folder
        self._frames = []
        self._pose: Optional[Pose] = None
        self._load()

    def _validate(self, frames: List[dict]):
        current_ts = 0
        for frame in frames:
            assert frame["timestamp"] > current_ts
            current_ts = frame["timestamp"]

    def _collect_frames(self, dict_frames: List[dict]):
        history_frames = []
        annotations_dict = json.load(
            open(os.path.join(self._scene_folder, "LIDAR_TOP", "annotations.json"))
        )
        for dict_frame in dict_frames:
            if dict_frame["is_key_frame"]:
                if self._pose is None:
                    raise ValueError("pose is not loaded!")
                frame = Frame(
                    self._raw_data_folder,
                    dict_frame,
                    history_frames,
                    annotations_dict.get(dict_frame["filename"], []),
                    self._pose,
                )
                history_frames = []
                self._frames.append(frame)
            else:
                history_frames.append(dict_frame)

    def _load(self):
        frame_file = os.path.join(self._scene_folder, "LIDAR_TOP", "frames.json")
        frames_data = json.load(open(frame_file, "r"))
        frames = frames_data["frames"]

        self._validate(frames)
        self._pose = Pose(
            quat=np.array(frames_data["calibrated_sensor"]["rotation"]),
            t=np.array(frames_data["calibrated_sensor"]["translation"]),
        )
        self._collect_frames(frames)

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, index: int):
        return self._frames[index]


class Scenes:
    def __init__(self, scene_indices: List[int]) -> None:
        self._scenes = []
        self._len = 0
        self._idx_to_scene_idx = dict()
        self._scenes_start_idx = []
        self._load_scenes(scene_indices)

    def __len__(self):
        return self._len

    def __getitem__(self, index: int):
        # return self._scenes[index]
        scene_idx = self._idx_to_scene_idx[index]
        frame = self._scenes[scene_idx][index - self._scenes_start_idx[scene_idx]]
        points = frame.get_points()
        annotations = frame.get_annotations(points, 3)
        return points, annotations

    def _load_scenes(self, indices: List[int]):
        data_root = "/mnt/data/datasets/nuScenes/"
        scene_idx = 0
        for index in indices:
            # example: "/home/bruce/work/perception/output/generated/nuScenes/scene-0009"
            scene_folder = os.path.join(
                QS_METADATA_ROOT,
                f"scene-{index:04d}",
            )
            print(scene_folder)
            scene = Scene(data_root, scene_folder)
            self._scenes.append(scene)
            self._scenes_start_idx.append(self._len)

            for i in range(len(scene)):
                self._idx_to_scene_idx[self._len + i] = scene_idx

            scene_idx += 1
            self._len += len(scene)


if __name__ == "__main__":

    MISSING_SCENE_IDS = set(
        json.load(
            open(
                "qs_datasets/nuscenes/lidar/MISSING_SCENES.json"
            )
        )["missing_ids"]
    )

    train_indices = list(set(range(1, 1000)) - MISSING_SCENE_IDS)

    scenes = Scenes(train_indices)
    print(len(scenes))

    import open3d as o3d
    from scipy.spatial.transform import Rotation
    import random
    print(len(scenes))
    for i in range(100):
        idx = random.randint(0, len(scenes))

        print(idx)
        points, annotations = scenes[idx]
        print(points.shape)
        print(annotations.shape)


        geometry = []
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        geometry.append(pcd)
        for annotation in annotations:
            # if annotation[4] < 10:
            #     continue
            print(annotation)
            box = o3d.geometry.OrientedBoundingBox(
                center=annotation[:3],
                R=Rotation.from_euler("xyz", [0, 0, annotation[6]]).as_matrix(),
                extent=annotation[3:6],
            )
            color_map = {
                0: (0, 0, 0),
                1: (1, 0, 0),
                2: (0, 1, 0),
                3: (0, 0, 1),
                4: (1, 1, 0),
            }
            box.color = color_map[annotation[7]]
            geometry.append(box)
        o3d.visualization.draw_geometries(geometry)








    # data_root = "/mnt/data/datasets/nuScenes/"
    # scene_folder = os.path.join(
    #     "/home/bruce/work/perception/output/generated/nuScenes/scene-0123"
    # )
    # scene = Scene(data_root, scene_folder)

    # import open3d as o3d
    # from scipy.spatial.transform import Rotation
    # import random
    # print(len(scene))
    # idx = random.randint(0, len(scene))
    # idx = 15
    # print(idx)
    # frame = scene[idx]

    # points = frame.get_points(10)
    # annotations = frame.get_annotations(points, 3)

    # geometry = []
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    # geometry.append(pcd)
    # for annotation in annotations:
    #     # if annotation[4] < 10:
    #     #     continue
    #     print(annotation)
    #     box = o3d.geometry.OrientedBoundingBox(
    #         center=annotation[:3],
    #         R=Rotation.from_euler("xyz", [0, 0, annotation[6]]).as_matrix(),
    #         extent=annotation[3:6],
    #     )
    #     color_map = {
    #         1: (1, 0, 0),
    #         2: (0, 1, 0),
    #         3: (0, 0, 1),
    #         4: (1, 1, 0),
    #     }
    #     box.color = color_map[annotation[7]]
    #     geometry.append(box)
    # o3d.visualization.draw_geometries(geometry)
