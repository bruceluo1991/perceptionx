"""
content in sample_data.json
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

"""

# 1. Given a pcd file name, find sample_token, calibrated_sensor_token and ego_pose_token from sample_data.json
# 2. By the sample_token, get object annotations from sample_annotation.json, lidar points must > 0
# 3. By the ego_pose_token and calibrated_sensor_token, get pose from ego_pose and sensor_pose, then transform the annotations to the lidar coord

import json
import os
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation

DATA_ROOT = "/mnt/data/datasets/nuScenes"
JSON_FOLDER = os.path.join(DATA_ROOT, "v1.0-trainval")
SAMPLE_DATA_FILE = os.path.join(JSON_FOLDER, "sample_data.json")
SAMPLE_ANNOTATION_FILE = os.path.join(JSON_FOLDER, "sample_annotation.json")
FULL_POSE_FILE = os.path.join(JSON_FOLDER, "ego_pose.json")

OUTPUT_FOLDER = "./output"

GENERATED_FOLDER = os.path.join(OUTPUT_FOLDER, "generated")
if not os.path.exists(GENERATED_FOLDER):
    os.makedirs(GENERATED_FOLDER)
LIDAR_SAMPLE_DATA_FILE = os.path.join(GENERATED_FOLDER, "lidar_sample_data.json")
LIDAR_SAMPLE_ANNOTATION_FILE = os.path.join(
    GENERATED_FOLDER, "lidar_sample_annotation.json"
)
LIDAR_SAMPLE_POSE_FILE = os.path.join(GENERATED_FOLDER, "lidar_sample_pose.json")
LIDAR_GEN_ANNOTATION_FOLDER = os.path.join(GENERATED_FOLDER, "annotation")
if not os.path.exists(LIDAR_GEN_ANNOTATION_FOLDER):
    os.makedirs(LIDAR_GEN_ANNOTATION_FOLDER)
LIDAR_GEN_ANNOTATION_EGO_COORD_FOLDER = os.path.join(
    GENERATED_FOLDER, "annotation_ego_coord"
)

if not os.path.exists(LIDAR_GEN_ANNOTATION_EGO_COORD_FOLDER):
    os.makedirs(LIDAR_GEN_ANNOTATION_EGO_COORD_FOLDER)


# 获取载入的pcd信息
def get_lidar_sample_data():
    if not os.path.exists(LIDAR_SAMPLE_DATA_FILE):
        # load json file
        with open(SAMPLE_DATA_FILE, "r", encoding="utf8") as f:
            sample_data = json.load(f)

        lidar_sample_data = []
        for item in sample_data:
            filename: str = item["filename"]
            if filename.startswith("samples") and "LIDAR_TOP" in filename:
                lidar_sample_data.append(item)

        with open(LIDAR_SAMPLE_DATA_FILE, "w", encoding="utf8") as f:
            json.dump(lidar_sample_data, f, indent=4)
    else:
        with open(LIDAR_SAMPLE_DATA_FILE, "r", encoding="utf8") as f:
            lidar_sample_data = json.load(f)
    return lidar_sample_data


# 获取sample数据的定位
def get_sample_pose_data():
    if not os.path.exists(LIDAR_SAMPLE_POSE_FILE):
        lidar_data = get_lidar_sample_data()
        ego_tokens = set()
        lidar_pose_data = dict()
        for item in lidar_data:
            ego_tokens.add(item["ego_pose_token"])
        with open(FULL_POSE_FILE, "r", encoding="utf8") as f:
            full_pose_data = json.load(f)
        for item in full_pose_data:
            if item["token"] in ego_tokens:
                lidar_pose_data[item["token"]] = item
        with open(LIDAR_SAMPLE_POSE_FILE, "w", encoding="utf8") as f:
            json.dump(lidar_pose_data, f, indent=4)
    else:
        with open(LIDAR_SAMPLE_POSE_FILE, "r", encoding="utf8") as f:
            lidar_pose_data = json.load(f)
    return lidar_pose_data


# 获取标注
def get_annotation_data():
    with open(SAMPLE_ANNOTATION_FILE, "r", encoding="utf8") as f:
        sample_annotation_data = json.load(f)
    sample_dict = defaultdict(list)
    for item in sample_annotation_data:
        sample_dict[item["sample_token"]].append(item)
    del sample_annotation_data
    return sample_dict


def generate_world_annotations():
    lidar_sample_data = get_lidar_sample_data()
    annotation_data = get_annotation_data()

    for item in lidar_sample_data:
        sample_token = item["sample_token"]
        if not sample_token in annotation_data:
            continue
        filename = item["filename"]
        base_name = os.path.basename(filename).split(".")[0]
        save_file_name = f"{base_name}.json"
        save_anno_file = os.path.join(LIDAR_GEN_ANNOTATION_FOLDER, save_file_name)
        with open(save_anno_file, "w", encoding="utf8") as f:
            json.dump(annotation_data[sample_token], f, indent=4)


def get_one_world_annotation_data(pcd_name: str):
    anno_file = os.path.join(LIDAR_GEN_ANNOTATION_FOLDER, pcd_name + ".json")
    if not os.path.exists(anno_file):
        return None
    with open(anno_file, "r", encoding="utf8") as f:
        world_annotation = json.load(f)
    return world_annotation
    # for item in items:
    #     quat = np.array(item["rotation"])
    #     translation = np.array(item["translation"])


def generate_local_annotations():
    lidar_sample_data = get_lidar_sample_data()
    annotation_data = get_annotation_data()

    for item in lidar_sample_data:
        sample_token = item["sample_token"]
        if not sample_token in annotation_data:
            continue
        filename = item["filename"]
        base_name = os.path.basename(filename).split(".")[0]
        save_file_name = f"{base_name}.json"
        save_anno_file = os.path.join(LIDAR_GEN_ANNOTATION_FOLDER, save_file_name)
        with open(save_anno_file, "w", encoding="utf8") as f:
            json.dump(annotation_data[sample_token], f, indent=4)


class Transform:

    def __init__(self, r: Rotation, t: np.ndarray):
        assert t.shape == (3,)
        self.r = r
        self.t = t


def transform_from_world_to_ego(
    p_world: np.ndarray,
    r_world: np.ndarray,
    r_ego: np.ndarray,
    t_ego: np.ndarray,
):
    rotation_imu_to_world: Rotation = Rotation.from_quat(r_ego, scalar_first=True)
    p_point_from_imu_world = p_world - t_ego
    p_point_from_imu_imu = rotation_imu_to_world.inv().apply(p_point_from_imu_world)
    rotation_point_to_world: Rotation = Rotation.from_quat(r_world, scalar_first=True)
    rotation_point_to_imu = rotation_imu_to_world.inv() * rotation_point_to_world
    theta = rotation_point_to_imu.as_euler("zyx", degrees=False)[0]
    return p_point_from_imu_imu, theta


def world_annotation_generated() -> bool:
    if not os.path.exists(LIDAR_GEN_ANNOTATION_FOLDER):
        return False
    files = os.listdir(LIDAR_GEN_ANNOTATION_FOLDER)
    if len(files) == 0:
        return False
    return True


def test_transform():
    p_w = np.array([100, 100, 0])
    r_w = np.array([1, 0, 0, 0])
    r_ego = np.array([0.5**2, 0, 0, 0.5**2])
    t_ego = np.array([0, 0, 0])
    print(transform_from_world_to_ego(p_w, r_w, r_ego, t_ego))


def generate_local_annotations():

    if not world_annotation_generated():
        generate_world_annotations()

    lidar_datas = get_lidar_sample_data()
    ego_poses = get_sample_pose_data()

    for item in lidar_datas:
        filename = item["filename"]
        # print(filename)
        base_name = os.path.basename(filename).split(".")[0]
        pose_token = item["ego_pose_token"]
        world_anntations = get_one_world_annotation_data(base_name)
        if not world_anntations:
            continue
        ego_pose = ego_poses[pose_token]
        for world_anntation in world_anntations:
            r_w = np.array(world_anntation["rotation"])
            p_w = np.array(world_anntation["translation"])
            r_ego = np.array(ego_pose["rotation"])
            t_ego = np.array(ego_pose["translation"])
            trans_local, theta = transform_from_world_to_ego(p_w, r_w, r_ego, t_ego)
            print(trans_local, theta)
            world_anntation["translation_ego"] = list(trans_local)
            world_anntation["theta_ego"] = theta
        with open(
            os.path.join(LIDAR_GEN_ANNOTATION_EGO_COORD_FOLDER, base_name + ".json"),
            "w",
            encoding="utf8",
        ) as f:
            json.dump(world_anntations, f, indent=4)
        break

generate_local_annotations()
