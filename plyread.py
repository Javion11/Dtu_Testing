"""
read a .ply file, return a 3*n np.ndarray [vertex_x, wertex_y, vertex_z]
"""
from plyfile import PlyData,PlyElement
import numpy as np 


def plyread(file_dir: str) -> np.array:
    with open(file_dir, 'rb') as f:
        plydata = PlyData.read(f) # 读取文件
        vertex_x = plydata['vertex']['x']
        vertex_y = plydata['vertex']['y']
        vertex_z = plydata['vertex']['z']
    return [vertex_x,vertex_y,vertex_z]

# test code, please ignore it
if __name__ == "__main__":
    plydata = PlyData.read("/algo/algo/hanjiawei/mvs_result/baseline_traineval_noise/baseline_noise4/mvsnet118_l3.ply")
    vertex_x = plydata['vertex']['x']
    print(len(vertex_x))
