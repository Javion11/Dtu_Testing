import numpy as np

def reducePts(pts, dst):
    """
    reduces a point set, pts, in a stochastic manner, such that the minmum sdistance ibetween points is 'dst'.
    """
    npoints = pts.shape[1]
    in

def PointCompareMain(cSet: int, Qdata: np.ndarray, dst: float, dataPath: str):
    # reduce points 0.2 mm neighbbourhood density
    Qdata = reducePts(Qdata,dst)
