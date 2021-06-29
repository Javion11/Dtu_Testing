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
