import os 
import time
import numpy as np
import scipy.io as scio
import multiprocess as mp

from plyread import plyread
from PointCompareMain import PointCompareMain

# create multiprocess subroutine to accelerate
def subroutine(params):
    datapath = params[0]
    plyPath = params[1]
    resultPath = plyPath
    results_txt = plyPath + "result.txt"

    method_string = "mvsnet"
    light_string = "l3" # l3 all lights on, l7 is randomly sampled between the 7 settings(index 0-6)
    representation_string = "Points"  # mvs representation "Points" or "Surfaces"

    if representation_string == "Points":
        eval_string = "_Eval_" # result naming
        # setting_string = ""
    dst = 0.2 # min dist between points when reducing
    
    cSet = params[2]
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
        
        print("Saving" + " scan" + cSet + " results")
        BaseEval_dict = BaseEval.__dict__
        scio.savemat(EvalName, BaseEval_dict)

        acc_mean = np.mean(BaseEval.FilteredDdata)
        acc_median = np.median(BaseEval.FilteredDdata)
        comp_mean = np.mean(BaseEval.FilteredDstl)
        comp_median = np.median(BaseEval.FilteredDstl)
        print("scan{} mean/median Data (acc.) {:.4f} / {:.4f}" .format(int(cSet), acc_mean, acc_median))
        print("scan{} mean/median Stl (comp.) {:.4f} / {:.4f}" .format(int(cSet), comp_mean, comp_median))

        # write result data to txtfile
        if not os.path.exists(results_txt):
            os.system("touch {}".format(results_txt))
        f = open(results_txt, "a")
        f.write("scan{} acc_mean: {:.4f}, acc_median: {:.4f}, comp_mean: {:.4f}, comp_median: {:.4f}\n"
                .format(int(cSet), acc_mean, acc_median, comp_mean, comp_median))