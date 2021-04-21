import time
import numpy as np
from sklearn.neighbors import KDTree
import scipy.io as scio

from plyread import plyread
from MaxDistCP import MaxDistCP

def reducePts(pts: np.ndarray, dst: float) -> np.ndarray:
    """
    reduces a point set, pts, in a stochastic manner, such that the minmum sdistance ibetween points is 'dst'.
    """
    nPoints = pts.shape[0]
    indexSet = np.full((nPoints,1), True, dtype=bool)
    RandOrd =  np.arange(nPoints)
    np.random.shuffle(RandOrd)

    # according to the points to create kd tree
    pts_random = np.random.permutation(pts)
    # test time used to create kdtree
    t1 = time.time()

    NS = KDTree(pts)
    
    t1_1 = time.time()
    print(float(t1_1-t1))

    # search the KTtree for close neighbours in a chunk-wise fashion to save memory if points cloud is really big
    Chunks = np.arange(0,nPoints,min(4e06, nPoints-1)).astype("int")
    for cChunk in range(len(Chunks)):
        if cChunk == len(Chunks) - 1:
            pts_to_search = pts_random[Chunks[cChunk]:, :]
        else:
            pts_to_search = pts_random[Chunks[cChunk]:Chunks[cChunk + 1], :]
        t2 = time.time()
        idx_points = NS.query_radius(pts_to_search, dst) # the point inputing in the search is only one object
        ptsOut = pts[idx_points[0]] # save points to output
        for idx_point in idx_points[1:]:
            ptsOut = np.vstack((ptsOut,pts[idx_point]))
    ptsOut = np.unique(ptsOut, axis=0)
    return ptsOut

def findByRow(mat: np.ndarray, raw: np.ndarray) -> np.ndarray:
    return np.where((mat == raw).all(1))[0]

class PointCompareMain():
    # def __init__(self, cSet: int, Qdata: np.ndarray, dst: float, dataPath: str, MaxDist=60: int):
    def __init__(self, cSet, Qdata, dst, dataPath, MaxDist=60):
        self.DataInMask = None
        self.dataPath = dataPath
        self.cSet = cSet 
        self.Margin = None # Margin of masks
        self.dst = dst # min dist between points when reducing
        self.Qdata = Qdata # input data points
        self.Ddata = None # distance from data to stl
        self.Qstl = None # input stl points
        self.Dstl = None # distance from stl to data
        self.MaxDist = MaxDist

        self.GroundPlane = None # Plane used to destingusie which stl points are 
        self.StlAbovePlane = None # Judge whether stl is above "ground plane"
        self.Time = None
        
        # use the plane and mask condition to filter the points
        self.FilteredDstl = None 
        self.FilteredDdata = None

        self.PointCompareMain(self.cSet, self.Qdata, self.dst, self.dataPath)

        

    def PointCompareMain(self, cSet: int, Qdata: np.ndarray, dst: float, dataPath: str):
        # reduce points 0.2 mm neighbbourhood density
        # if points numbers in Qdata is smaller than 4e06, skip the reduce Points scale step
        if len(Qdata) > 4e06:
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
        ObsMask_shape = np.array(ObsMask.shape)

        MaxDist_default = 60
        print("sacn" + cSet + " Computing Data to Stl distances")
        # Ddata = MaxDistCP(Qstl, Qdata, BB, MaxDist_default)
        Ddata = MaxDistCP(Qstl, Qdata, MaxDist_default)

        print("sacn" + cSet + " Computing Stl to Data distances")
        # Dstl = MaxDistCP(Qdata, Qstl, BB, MaxDist_default)
        Dstl = MaxDistCP(Qdata, Qstl, MaxDist_default)
        print("sacn" + cSet + " Distances computed")

        # use mask
        # from get mask - inverted & modified
        one = np.ones((Qdata.shape[0], 1))
        Qv = (Qdata - one * BB[0,:]) / Res + 1
        Qv = np.round(Qv)
        Qv = Qv.astype("int")

        # remove the points out of region(ObsMask)
        self.DataInMask = np.full((Qv.shape[0], 1), False, dtype=bool)
        Midx_list = [] # Midx refers to the point's index
        for Midx, Qv_pt in enumerate(Qv):
            if np.all(Qv_pt > 0) and np.all(Qv_pt < ObsMask_shape ):
                    Midx_list.append(Midx)
                    if ObsMask[Qv_pt[0]][Qv_pt[1]][Qv_pt[2]] == 1:
                        self.DataInMask[Midx] = True     
        
        """
        another method to build DataInMask: ulitize set to find true index, unfortunately don't work,
        because the set will remove the repeat elements
        """
        # time_mask_1 = time.time()
        # self.DataInMask = np.full((Qv.shape[0], 1), True, dtype=bool)
        # ObsMask_true_index = np.where(ObsMask == 1)
        # # ObsMask_true_index = np.transpose(ObsMask_true_index)
        # ObsMask_true_index_set = set(zip(*ObsMask_true_index))
        # Qv_set = set(zip(*np.transpose(Qv)))
        # Qv_set_Flase = Qv_set - ObsMask_true_index_set
        # for element in Qv_set_Flase:
        #     element = np.array(list(element))
        #     element_index = findByRow(Qv, element)
        #     self.DataInMask[element_index] = False
        # time_mask_2 = time.time()
        # print(time_mask_2 - time_mask_1)


        self.Margin = Margin 
        self.Ddata = Ddata 
        self.Qstl = Qstl
        self.Dstl = Dstl

        plane_data_path = dataPath + "ObsMask/Plane" + cSet +".mat"
        plane_data = scio.loadmat(plane_data_path)
        self.GroundPlane = plane_data['P']
        self.StlAbovePlane = np.dot(np.hstack((Qstl,np.ones((Qstl.shape[0],1)))), plane_data['P']) > 0
        self.Time = time.time()

        MaxDist = self.MaxDist
        self.FilteredDdata = self.Ddata[self.DataInMask]
        self.FilteredDdata = self.FilteredDdata[self.FilteredDdata < self.MaxDist]
        self.FilteredDstl = self.Dstl[self.StlAbovePlane]
        self.FilteredDstl = self.FilteredDstl[self.FilteredDstl < self.MaxDist]



