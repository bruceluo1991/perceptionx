# 数据划分
- 在最上层，按照场景进行划分
- 每个场景下，按照main frame进行划分

## Attention

<font color="red">nuscenes数据集的第五维度不是时间，而是ring id, ring 代表激光雷达的不同通道</font>

## 数据格式
- All
    - Scene
        - File: Scene Description json file
        - Folder: Lidar 1
            - File: frames.json
                - Content:
                    - pcd file
                    - is key frame
                    - ego pose
                    - timestamp
            - File: Annotation.json
                - Content: Annotation json
                    - key: main frame annotation
        - Folder: Lidar 2
        - Folder: Camera 1
        - Folder: Camera 2
        - Folder: Radar 1
        - Folder: Radar 2

### Pose
### Annotation

## 代码
### 数据生成
- 生成数据格式中的数据
    - 生成Scene Description json file
    - 每个Scene下，生成Lidar文件夹
        - Lidar.json
        - 生成Pose.json
        - 生成Annotation.json

### 数据读取
- Dataset
    - Scene
