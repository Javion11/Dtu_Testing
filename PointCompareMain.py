import numpy as np
import kdtree
import scipy.io as scio

from plyread import plyread

def reducePts(pts, dst):
    """
    reduces a point set, pts, in a stochastic manner, such that the minmum sdistance ibetween points is 'dst'.
    """
    nPoints = pts.shape[0]
    indexSet = np.full((nPoints,1), True, dtype=bool)
    RandOrd =  np.arange(nPoints)
    np.random.shuffle(RandOrd)

    # according to the points to create kd tree
    pts_random = np.random.permutation(pts)
    pts_random = list(pts_random)
    pts = list(pts)
    NS = kdtree.create(pts)

    # search the KTtree for close neighbours in a chunk-wise fashion to save memory if points cloud is really big
    ptsOut = [] # save points to output

    Chunks = np.arange(0,nPoints,min(4e06, nPoints-1))
    for cChunk in range(len(Chunks)):
        pts_to_search = pts_random[Chunks[cChunk]:Chunks[cChunk + 1] + 1]
        for pt_to_search in pts_to_search:
            """
            Search the n nearest nodes of the given point which are within given distance,point must be a location, not a node.
            A list containing the n nearest nodes to the point within the distance will be returned.
            Note:the point inputing in the search is only one object.
            """
            idx_points = NS.search_nn_dist(pt_to_search, dst) # the point inputing in the search is only one object
            for idx_point in idx_points:
                if idx_point not in ptsOut:
                    ptsOut.append(idx_point)
    print("downsample factor: {:.4f}".format(nPoints/len(ptsOut)))    
    


class PointsCompareMain():
    def __init__(self, cSet: int, Qdata: np.ndarray, dst: float, dataPath: str):
        self.DataInMask = None
        self.Cset = cSet 
        self.Margin = None # Margin of masks
        self.dst = dst # min dist between points when reducing
        self.Qdata = Qdata # input data points
        self.Ddata = None # distance from data to stl
        self.Qstl = None # input stl points
        self.Dstl = None # distance from stl to data
        self.PointCompareMain(self.Cset, self.Qdata, self.dst, self.dataPath)

    def PointCompareMain(self, cSet: int, Qdata: np.ndarray, dst: float, dataPath: str):
        # reduce points 0.2 mm neighbbourhood density
        Qdata = reducePts(Qdata,dst)
        cSet = str(cSet)
        cSet_fill = cSet.zfill(3)
        StlInName = dataPath + 'Points/stl/stl' + cSet_fill + '_total.ply'

        StlMesh = plyread(StlInName) # stl points already reduced 0.2mm neighbourhood density
        Qstl = np.transpose(StlMesh) 

        # load mask (ObsMask) and bounding box and resolution (Res)
        Margin = 10
        MaskName = dataPath + 'ObsMask/ObsMask' + cSet + '_' + str(Margin) + '.mat'
        mask_data = scio.loadmat(MaskName) # use scipy.io.loadmat to read *.mat file, return a python dict 
        BB = mask_data['BB']
        Res = mask_data['Res']
        ObsMask = mask_data['ObsMask']

        MaxDist = 60
        print("Computing Data to Stl distances")
        Ddata = MaxDistCP(Qstl, Qdata, BB, MaxDist)

        print("Computing Stl to Data distances")
        Dstl = MaxDistCP(Qdata, Qstl, BB, MaxDist)
        print("Distances computed")

        # use mask
        # from get mask - inverted & modified
        one = np.ones((Qdata.shape[0], 1))
        Qv = (Qdata - one * BB[0,:]) / Res + 1
        Qv = np.round(Qv)

        # remove the points out of region(ObsMask)
        self.DataInMask = np.full((Qv.shape[0], 1), value=False, dtype=bool)
        Midx_list = [] # Midx1 refers to the point's index
        # MidxA_list = [] # MidxA refers to the point in axis's location 
        for Midx, Qv_pt in enumerate(Qv):
            if Qv_pt[0] > 0 and Qv_pt[0] <= ObsMask.shape[0] and\
                Qv_pt[1] > 0 and Qv_pt[1] <= ObsMask.shape[1] and\
                    Qv_pt[2] > 0 and Qv_pt[2] <= ObsMask.shape[2]:
                    Midx1_list.append(Midx1)
                    if ObsMask(Qv_pt[0], Qv_pt[1], Qv_pt[2]) == 1:
                        self.DataInMask[Midx] = True
        self.Margin = Margin # Margin of masks
        self.Ddata = Ddata 
        self.Qstl = Qstl
        self.Dstl = Dstl





        


# test code, don't mind 
if __name__ == "__main__":
    Margin = 10
    cSet = "1"
    dataPath = "/algo/algo/hanjiawei/DataSet/SampleSet/MVS Data/"
    MaskName = dataPath + 'ObsMask/ObsMask' + cSet + '_' + str(Margin) + '.mat'
    data_mat = scio.loadmat(MaskName)
    print("test!")