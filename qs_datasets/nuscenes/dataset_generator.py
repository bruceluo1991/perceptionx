import os
import json
from collections import defaultdict
from enum import IntEnum

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Optional, Set, Dict, List, Any


DATA_ROOT = "/mnt/data/datasets/nuScenes/"
OUTPUT_ROOT = "output/generated/nuScenes/"

IN_ROOT_FOLDER = os.path.join(DATA_ROOT, "v1.0-trainval")
IN_CALIB_SENSOR_FILE = "calibrated_sensor.json"
IN_SAMPLE_FILE = "sample.json"
IN_SAMPLE_DATA_FILE = "sample_data.json"
IN_POSE_FILE = "ego_pose.json"
IN_INSTANCE_FILE = "instance.json"
IN_ANNOTATION_FILE = "sample_annotation.json"
IN_CATEGORY_FILE = "category.json"

OUT_DESCRIPTION_FILE = "description.json"


def get_input_root_folder():
    return IN_ROOT_FOLDER


class ObstacleType(IntEnum):
    Reserve = 0  # "reserve"
    Vehicle = 1  # "vehicle"
    Cycle = 2  # "cycle"
    Human = 3  # "human"
    Misc = 4  # "misc"


MAPPING = {
    "human.pedestrian.adult": ObstacleType.Human,
    "human.pedestrian.child": ObstacleType.Human,
    "human.pedestrian.wheelchair": ObstacleType.Cycle,
    "human.pedestrian.stroller": ObstacleType.Cycle,
    "human.pedestrian.personal_mobility": ObstacleType.Cycle,
    "human.pedestrian.police_officer": ObstacleType.Human,
    "human.pedestrian.construction_worker": ObstacleType.Human,
    "animal": ObstacleType.Misc,
    "vehicle.car": ObstacleType.Vehicle,
    "vehicle.motorcycle": ObstacleType.Cycle,
    "vehicle.bicycle": ObstacleType.Cycle,
    "vehicle.bus.bendy": ObstacleType.Vehicle,
    "vehicle.bus.rigid": ObstacleType.Vehicle,
    "vehicle.truck": ObstacleType.Vehicle,
    "vehicle.construction": ObstacleType.Vehicle,
    "vehicle.emergency.ambulance": ObstacleType.Vehicle,
    "vehicle.emergency.police": ObstacleType.Vehicle,
    "vehicle.trailer": ObstacleType.Vehicle,
    "movable_object.barrier": ObstacleType.Misc,
    "movable_object.trafficcone": ObstacleType.Misc,
    "movable_object.pushable_pullable": ObstacleType.Misc,
    "movable_object.debris": ObstacleType.Misc,
    "static_object.bicycle_rack": ObstacleType.Reserve,
}


class CategoryManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):

        if self._initialized:
            return
        file_path = os.path.join(get_input_root_folder(), IN_CATEGORY_FILE)
        with open(file_path, "r", encoding="utf8") as f:
            self._categories = json.load(f)
        self._category_dict = {}
        for category in self._categories:
            self._category_dict[category["token"]] = category
        self._initialized = True

    @property
    def category_dict(self):
        return self._category_dict


class InstanceManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        file_path = os.path.join(get_input_root_folder(), IN_INSTANCE_FILE)
        with open(file_path, "r", encoding="utf8") as f:
            self._instances = json.load(f)
        self._token_instance_dict = {
            instance["token"]: instance for instance in self._instances
        }
        self._initialized = True

    @property
    def instance_dict(self):
        return self._token_instance_dict


class SampleDataManager:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        file_path = os.path.join(get_input_root_folder(), IN_SAMPLE_DATA_FILE)
        with open(file_path, "r", encoding="utf8") as f:
            sample_data = json.load(f)

        self._token_data_dict: Dict[str, Any] = {}
        self._sample_data_dict: Dict[str, List[Any]] = defaultdict(list)
        self._sample_data_sensor_dict: Dict[str, Dict[str, Any]] = defaultdict(dict)

        for data in sample_data:
            sensor_name = data["filename"].split("/")[1]
            sample_token = data["sample_token"]
            self._token_data_dict[data["token"]] = data
            self._sample_data_dict[sample_token].append(data)
            if data["is_key_frame"]:
                self._sample_data_sensor_dict[sample_token][sensor_name] = data

        self._initialized = True

    # === 全量数据属性 ===
    @property
    def token_data_dict(self):
        return self._token_data_dict

    @property
    def sample_data_dict(self):
        return self._sample_data_dict

    @property
    def sample_data_sensor_dict(self):
        return self._sample_data_sensor_dict


class SampleAnnotationManager:
    def __init__(self):
        if hasattr(self, "_annotation_dict"):
            return
        file_path = os.path.join(get_input_root_folder(), IN_ANNOTATION_FILE)
        with open(file_path, "r", encoding="utf8") as f:
            self._annotations = json.load(f)
        self._annotation_dict = {}
        self._sample_dict = defaultdict(list)
        for annotation in self._annotations:
            self._annotation_dict[annotation["token"]] = annotation
            sample_token = annotation["sample_token"]
            self._sample_dict[sample_token].append(annotation)

    @property
    def annotation_dict(self):
        return self._annotation_dict

    @property
    def sample_dict(self):
        return self._sample_dict

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance


class PoseManager:
    def __init__(self):
        if hasattr(self, "_pose_dict"):
            return
        file_path = os.path.join(get_input_root_folder(), IN_POSE_FILE)
        with open(file_path, "r", encoding="utf8") as f:
            self._poses = json.load(f)
        self._pose_dict = {}
        for pose in self._poses:
            self._pose_dict[pose["token"]] = pose

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def pose_dict(self):
        return self._pose_dict


class SampleManager:
    def __init__(self):
        if hasattr(self, "_samples"):
            return
        file_path = os.path.join(get_input_root_folder(), IN_SAMPLE_FILE)
        with open(file_path, "r", encoding="utf8") as f:
            self._samples = json.load(f)
        scene_samples_dict = defaultdict(list)
        for sample in self._samples:
            scene_samples_dict[sample["scene_token"]].append(sample)
        # sort by timestamp
        for scene_samples in scene_samples_dict.values():
            scene_samples.sort(key=lambda x: x["timestamp"])
        self._scene_samples_dict = scene_samples_dict

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def scene_samples_dict(self):
        return self._scene_samples_dict


# class CalibratedSensor:
#     def __init__(self, dict_data: Dict):
#         self._rotation = Rotation.from_quat(dict_data["rotation"], scalar_first=True)
#         self._translation = np.array(dict_data["translation"])

#     @property
#     def rotation(self):
#         return self._rotation

#     @property
#     def translation(self):
#         return self._translation


class CalibratedSensorManager:

    def __init__(self):
        if hasattr(self, "_sensor_dict"):
            return
        sensor_file_path = os.path.join(get_input_root_folder(), IN_CALIB_SENSOR_FILE)
        with open(sensor_file_path, "r", encoding="utf8") as f:
            self._dict_data = json.load(f)
        self._sensor_dict = dict()
        for sensor in self._dict_data:
            self._sensor_dict[sensor["token"]] = sensor

    @property
    def sensor_dict(self):
        return self._sensor_dict

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance


class LidarGenerator:
    def __init__(self, output_root: str, dict_data: Dict):
        self._output_root = output_root

    def generate(self):
        self._generate_frames_json()
        self._generate_pose_json()
        self._generate_annotation_json()

    def _generate_frames_json(self):
        pass

    def _generate_pose_json(self):
        pass

    def _generate_annotation_json(self):
        pass


class SceneGenerator:
    def __init__(self, output_root: str, dict_data: Dict):
        self._dict_data = dict_data
        self._name = self._dict_data["name"]
        self._token = self._dict_data["token"]
        self._output_dir = os.path.join(OUTPUT_ROOT, self._name)
        os.makedirs(self._output_dir, exist_ok=True)
        self._description_file = os.path.join(self._output_dir, OUT_DESCRIPTION_FILE)
        self._generate_description()

    def _generate_description(self):
        with open(self._description_file, "w", encoding="utf8") as f:
            json.dump(self._dict_data, f, indent=4)

    def _get_samples(self):
        return SampleManager().scene_samples_dict[self._token]

    def _generate_output_frame_data(self, sample_data: Dict) -> Dict:
        output_data = dict()
        output_data["filename"] = sample_data["filename"]
        output_data["timestamp"] = sample_data["timestamp"]
        output_data["is_key_frame"] = sample_data["is_key_frame"]
        output_data["ego_pose"] = dict()
        ego_pose = PoseManager().pose_dict[sample_data["ego_pose_token"]]
        output_data["ego_pose"]["translation"] = ego_pose["translation"]
        output_data["ego_pose"]["rotation"] = ego_pose["rotation"]
        return output_data

    def generate(self, sensor_name: str):
        sensor_root_path = os.path.join(self._output_dir, sensor_name)
        os.makedirs(sensor_root_path, exist_ok=True)
        self.generate_frames(sensor_name)
        self.generate_annotations(sensor_name)

    def generate_frames(self, sensor_name: str):
        samples = self._get_samples()
        result = {}

        sample_data_token_set = set()
        for idx, sample in enumerate(samples):
            sample_token = sample["token"]

            sample_data_manager = SampleDataManager()
            pose_manager = PoseManager()
            sample_data = sample_data_manager.sample_data_sensor_dict[sample_token][
                sensor_name
            ]
            this_token = sample_data["token"]
            calibrated_sensor_token = sample_data["calibrated_sensor_token"]
            # Making sure all sample data are in the set
            if idx > 0 and this_token != "":
                assert (
                    this_token in sample_data_token_set
                ), f"sample_token {this_token} not in sample_data_token_set"
                continue

            output_data_list = []
            while this_token != "":
                sample_data_token_set.add(this_token)
                sample_data = sample_data_manager.token_data_dict[this_token]
                assert calibrated_sensor_token == sample_data["calibrated_sensor_token"]
                output_data = self._generate_output_frame_data(sample_data)
                output_data_list.append(output_data)
                this_token = sample_data["next"]
            assert idx == 0
            result["frames"] = output_data_list
            result["calibrated_sensor"] = {
                "token": calibrated_sensor_token,
                "rotation": CalibratedSensorManager()
                .sensor_dict[calibrated_sensor_token]["rotation"],
                "translation": CalibratedSensorManager()
                .sensor_dict[calibrated_sensor_token]["translation"],
            }
        output_file = os.path.join(self._output_dir, sensor_name, "frames.json")
        with open(output_file, "w", encoding="utf8") as f:
            json.dump(result, f, indent=4)

    def generate_annotations(self, sensor_name: str):
        # annotation_folder = os.path.join(self._output_dir, sensor_name, "annotations")
        # os.makedirs(annotation_folder, exist_ok=True)
        samples = self._get_samples()
        annotation_manager = SampleAnnotationManager()
        sample_data_manager = SampleDataManager()
        output = defaultdict(list)
        for sample in samples:
            sample_token = sample["token"]
            sample_data = sample_data_manager.sample_data_sensor_dict[sample_token][
                sensor_name
            ]
            annotations = annotation_manager.sample_dict[sample_token]
            sample_sensor_file = sample_data["filename"]

            for annotation in annotations:
                instance_token = annotation["instance_token"]
                instance = InstanceManager().instance_dict[instance_token]
                category_token = instance["category_token"]
                category = CategoryManager().category_dict[category_token]
                obstacle_type = MAPPING[category["name"]]
                output[sample_sensor_file].append({
                    "obstacle_type": obstacle_type,
                    "translation": annotation["translation"],
                    "size": annotation["size"],
                    "rotation": annotation["rotation"],
                    "instance_token": instance_token,
                    "visibility": annotation["visibility_token"],
                    "num_lidar_pts": annotation["num_lidar_pts"],
                })

        output_file = os.path.join(self._output_dir, sensor_name, "annotations.json")
        with open(output_file, "w", encoding="utf8") as f:
            json.dump(output, f, indent=4)

            # print(annotation_manager.sample_dict[sample_token])
            # print(sample)
            # print(sample_data)
        
def _test():
    with open(os.path.join(IN_ROOT_FOLDER, "scene.json"), "r", encoding="utf8") as f:
        dict_datas = json.load(f)
    for dict_data in dict_datas:
        if dict_data["name"] != "scene-0123":
            continue
        scene_generator = SceneGenerator(OUTPUT_ROOT, dict_data)
        scene_generator.generate("LIDAR_TOP")

def _main():
    with open(os.path.join(IN_ROOT_FOLDER, "scene.json"), "r", encoding="utf8") as f:
        dict_datas = json.load(f)
    for dict_data in dict_datas:
        scene_generator = SceneGenerator(OUTPUT_ROOT, dict_data)
        scene_generator.generate("LIDAR_TOP")

if __name__ == "__main__":

    _main()
