from scipy.spatial.transform import Rotation
import numpy as np
from typing import Union

class Pose:
    def __init__(self, quat: Union[np.ndarray, Rotation], t: np.ndarray):
        if isinstance(quat, np.ndarray):
            assert quat.shape == (4,)
            self.r = Rotation.from_quat(quat, scalar_first=True)
        else:
            assert isinstance(quat, Rotation)
            self.r = quat
        self.t = t
    
    def __repr__(self) -> str:
        return f"Pose(quat={self.r.as_quat()}, t={self.t})"
    
    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.r.apply(p) + self.t

    def transform_inv(self, p: np.ndarray) -> np.ndarray:
        return self.r.inv().apply(p - self.t)
    
    def inv(self) -> "Pose":
        return Pose(self.r.inv(), -self.r.inv().apply(self.t))



def _test():
    p = Pose(np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]), np.array([1, 2, 3]))
    print(p)
    print(p.transform(np.array([1, 0, 0])))
    print(p.transform_inv(np.array([1, 0, 0])))
    print(p.inv())
    print(p.inv().transform(np.array([1, 0, 0])))
    print(p.inv().transform_inv(np.array([1, 0, 0])))


if __name__ == "__main__":
    _test()
