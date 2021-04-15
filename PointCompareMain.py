import numpy as np
import kdtree

def reducePts(pts, dst):
    """
    reduces a point set, pts, in a stochastic manner, such that the minmum sdistance ibetween points is 'dst'.
    """
    nPoints = pts.shape[1]
    indexSet = np.full((nPoints,1), True, dtype=bool)
    RandOrd =  np.arange(nPoints)
    np.random.shuffle(RandOrd)

    # according to the points to create kd tree
    NS = kdtree.create(np.transpose(pts))

    # search the KTtree for close neighbours in a chunk-wise fashion to save memory if points cloud is really big
    Chunks = np.arange(0,nPoints,min(4e06, nPoints-1))
    for Chunk in range(len(Chunks)):
        Range = np.arange(Chunks[Chunk],Chunks[Chunk + 1] + 1)

        """
        Search the n nearest nodes of the given point which are within given distance
        point must be a location, not a node. A list containing the n nearest nodes to the point within the distance will be returned.
        """
        idx = NS.search_nn_dist(NS, np.transpose(pts[:,RandOrd[Range]]), dst)

        # for i in range()




def PointCompareMain(cSet: int, Qdata: np.ndarray, dst: float, dataPath: str):
    # reduce points 0.2 mm neighbbourhood density
    Qdata = reducePts(Qdata,dst)
