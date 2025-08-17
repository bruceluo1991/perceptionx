import numpy as np

class Annotation:
    def __init__(self, data: dict):
        self.pointcloud:np.ndarray = data["pointcloud"] # N, 5: x,y,z,i,t
        self.annotations:np.ndarray = data["annotation"] # M, 8: x,y,z,w,l,h,yaw,type in ego coord.
