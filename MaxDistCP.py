import time
import numpy as np
from sklearn.neighbors import KDTree

def MaxDistCP(Qto: np.ndarray, Qfrom: np.ndarray, MaxDist: float) -> np.ndarray:
    Qto = np.array(Qto)
    Qfrom = np.array(Qfrom)
    kd_Qto = KDTree(Qto)
    dist = kd_Qto.query(Qfrom, k=1, return_distance=True)[0]
    boundary_bool = dist > MaxDist
    dist[boundary_bool] = MaxDist
    return dist



"""
'''
refer to matlab code, the logic is werid, so dorp it.
'''
# import kdtree
# from numba import jit

# @jit(nopython=True)
def MaxDistCP(Qto: np.ndarray, Qfrom: np.ndarray, BB: np.ndarray, MaxDist: float) -> np.ndarray:
    Dist = np.ones((Qfrom.shape[0],1)) * MaxDist
    Range = np.floor((BB[1]-BB[0]) / MaxDist)
    Done = 0 # record the number of points has been computed

    for x in range(int(Range[0])):
        for y in range(int(Range[1])):
            for z in range(int(Range[2])):
                t1 = time.time()
                # split points to the target region
                Low = BB[0] + np.array([x, y, z])*MaxDist
                High = Low + MaxDist
                SQfrom = [] # the points to be compared in target region 
                SQto = [] # the stl points in the target
                idxF_list = []
                for idxF,Qfrom_pt in enumerate(Qfrom):
                    if Qfrom_pt[0] >= Low[0] and Qfrom_pt[0] < High[0] and\
                        Qfrom_pt[1] >= Low[1] and Qfrom_pt[1] < High[1] and\
                        Qfrom_pt[2] >= Low[2] and Qfrom_pt[2] < High[2]:
                        SQfrom.append(Qfrom_pt)
                        idxF_list.append(idxF)
                
                Low = Low - MaxDist
                High = High + MaxDist
                for Qto_pt in Qto:
                    if Qto_pt[0] >= Low[0] and Qto_pt[0] < High[0] and\
                        Qto_pt[1] >= Low[1] and Qto_pt[1] < High[1] and\
                        Qto_pt[2] >= Low[2] and Qto_pt[2] < High[2]:
                        SQto.append(Qto_pt)
                
                if len(SQto) == 0:
                    Dist[idxF_list] = MaxDist
                else:
                    KDstl = kdtree.create(SQto)
                    for SQfrom_pt, idxF in zip(SQfrom, idxF_list):
                        # Search the nearest node of the given point
                        # The result is a (node, distance) tuple.
                        Dist[idxF] = KDstl.search_nn(SQfrom_pt)[1]
                Done = Done + len(idxF_list) 
                t1_2 = time.time()
                print(float(t1_2-t1))

"""

