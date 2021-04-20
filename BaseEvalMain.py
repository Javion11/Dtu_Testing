import os 
import time
import numpy as np
import scipy.io as scio
from multiprocessing import Pool

from plyread import plyread
from PointCompareMain import PointCompareMain

# to calculate distances have been measured for all included sacns(UsedSets)
datapath = "/algo/algo/hanjiawei/DataSet/SampleSet/MVS Data/" 
plyPath = "/algo/algo/hanjiawei/mvs_result/baseline_traineval_noise/baseline_noise2/"
resultPath = "/algo/algo/hanjiawei/mvs_result/baseline_traineval_noise/baseline_noise2/"

method_string = "mvsnet"
light_string = "l3" # l3 all lights on, l7 is randomly sampled between the 7 settings(index 0-6)
representation_string = "Points"  # mvs representation "Points" or "Surfaces"


if representation_string == "Points":
    eval_string = "_Eval_" # result naming
    setting_string = ""

# scans used to eval
UsedSets = [1,4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118]

dst = 0.2 # min dist between points when reducing

# create multiprocess to accelerate
def subroutine(param):
    cSet = param
    cSet = str(cSet)
    cSet_fill = cSet.zfill(3)
    plyname = method_string + cSet_fill + "_" + light_string + ".ply"

    DataInName = os.path.join(plyPath, plyname)
    EvalName = resultPath + method_string + eval_string + cSet + ".mat"
    
    # check eval result file is already computed
    if not os.path.exists(EvalName):
        print(EvalName)
        
        # read the ply file to eval
        Qdata = plyread(DataInName) 
        Qdata = np.transpose(Qdata)

        MaxDist = 20 # outlier threshold of 20mm
        BaseEval = PointCompareMain(cSet, Qdata, dst, datapath, MaxDist)
        
        print("Saving results")
        BaseEval_dict = BaseEval.__dict__
        scio.savemat(EvalName, BaseEval_dict)

        print("mean/median Data (acc.) {:.4f} / {:.4f}" .format(np.mean(BaseEval.FilteredDdata), np.median(BaseEval.FilteredDdata)))
        print("mean/median Stl (comp.) {:.4f} / {:.4f}" .format(np.mean(BaseEval.FilteredDstl), np.median(BaseEval.FilteredDstl)))

with Pool(16) as p:
    p.map(subroutine, UsedSets)

# for cSet in UsedSets:
#     cSet = str(cSet)
#     cSet_fill = cSet.zfill(3)
#     plyname = method_string + cSet_fill + "_" + light_string + ".ply"

#     DataInName = os.path.join(plyPath, plyname)
#     EvalName = resultPath + method_string + eval_string + cSet + ".mat"
    
#     # check eval result file is already computed
#     if not os.path.exists(EvalName):
#         print(EvalName)
        
#         # read the ply file to eval
#         Qdata = plyread(DataInName) 
#         Qdata = np.transpose(Qdata)

#         MaxDist = 20 # outlier threshold of 20mm
#         BaseEval = PointCompareMain(cSet, Qdata, dst, datapath, MaxDist)
        
#         print("Saving results")
#         BaseEval_dict = BaseEval.__dict__
#         scio.savemat(EvalName, BaseEval_dict)

#         print("mean/median Data (acc.) {:.4f} / {:.4f}" .format(np.mean(BaseEval.FilteredDdata), np.median(BaseEval.FilteredDdata)))
#         print("mean/median Stl (comp.) {:.4f} / {:.4f}" .format(np.mean(BaseEval.FilteredDstl), np.median(BaseEval.FilteredDstl)))


        

