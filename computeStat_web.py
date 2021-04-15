import os 
import time
import numpy as np

from plyread import plyread

# to calculate distances have been measured for all included sacns(UsedSets)
datapath = ""
plyPath = ""
resultPath = ""

method_string = "mvsnet"
light_string = "13" # l3 all lights on, l7 is randomly sampled between the 7 settings(index 0-6)
representation_string = "Points"  # mvs representation "Points" or "Surfaces"


if representation_string == "Points":
    eval_string = "_Eval_" # result naming
    setting_string = ""

# scans used to eval
UsedSets = [1,4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118]

dst = 0.2 # min dist between points when reducing

for cSet in UsedSets:
    cSet = str(cSet)
    cSet_fill = cSet.zfill(3)
    plyname = method_string + cSet_fill + "_" + light_string + ".ply"

    DataInName = os.path.join(plyPath, plyname)
    EvalName = resultPath + method_string + eval_string + cSet + ".mat"
    
    # check eval result file is already computed
    if not os.path.exists(EvalName):
        print(EvalName)
        
        # read the .ply file to eval
        Qdata = plyread(DataInName) 
        Qdata = np.transpose(Qdata)

        BaseEval = PointCompareMain(cSet, Qdata, dst, datapath)
        


