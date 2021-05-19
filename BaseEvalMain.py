import os
import argparse
import numpy as np
import multiprocess as mp
from itertools import product

from evalmain_subroutine import subroutine

parser = argparse.ArgumentParser(description="Eval ply distance between the generated and the stl")
parser.add_argument("--datapath", help="the path of stl ply file")
parser.add_argument("--plyPath", help="the path of the generated ply file")
args = parser.parse_args()
args.datapath = "/algo/algo/hanjiawei/DataSet/SampleSet/MVS Data/"
args.plyPath = "/algo/algo/hanjiawei/denoise_result/mvsnet_cascade_base/"

if __name__ == "__main__":
    datapath = [args.datapath]
    plyPath = [args.plyPath]
    UsedSets = [1,4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118] # scans used to eval
    params_list  = product(datapath, plyPath, UsedSets)
    # excute multiprocess
    with mp.get_context("spawn").Pool(22) as pool:
       pool.map(subroutine, params_list)
    
    # compute stat of results
    results_txt = args.plyPath + "result.txt"
    f_read = open(results_txt, "r")
    results_list = f_read.readlines()
    f_read.close()
    # some stat of data
    results_acc_mean = []
    results_acc_median = []
    results_comp_mean = []
    results_comp_median = []
    for results in results_list:
        single_result_list = results.split(",")
        for i, single_result in enumerate(single_result_list):
            if i == 0 :
                results_acc_mean.append(float(single_result.split(':')[1]))
            elif i == 1:
                results_acc_median.append(float(single_result.split(':')[1]))
            elif i == 2:
                results_comp_mean.append(float(single_result.split(':')[1]))
            elif i == 3:
                results_comp_median.append(float(single_result.split(':')[1]))
    results_acc_mean = np.mean(np.array(results_acc_mean))
    results_acc_median = np.mean(np.array(results_acc_median))
    results_comp_mean = np.mean(np.array(results_comp_mean))
    results_comp_median = np.mean(np.array(results_comp_median))

    f_write = open(results_txt, "a")
    f_write.write("all scans, acc_mean: {:.4f}, acc_median: {:.4f}, comp_mean: {:.4f}, comp_median: {:.4f}"
        .format(results_acc_mean, results_acc_median, results_comp_mean, results_comp_median))
    f_write.close()
    
    # method 2, this way will be slow
    # results = []
    # pool = mp.get_context("spawn").Pool(22)
    # for params in params_list:
    #     pool_result = pool.apply_async(subroutine, args=(params,))
    #     results.append(pool_result.get())
    # pool.close()
    # pool.join()
    
    # some stat of data
    # results_acc_mean = []
    # results_acc_median = []
    # results_comp_mean = []
    # results_comp_median = []

    # # write result data to txtfile
    # if not os.path.exists(results_txt):
    #     os.system("touch {}".format(results_txt))
    # f = open(results_txt, "a")
    # for result in results:
    #     f.write(str(result) + '\n')
    #     results_acc_mean.append(result[0])
    #     results_acc_median.append(result[1])
    #     results_comp_mean.append(result[2])
    #     results_comp_median.append(result[3])

    # results_acc_mean = np.array(results_acc_mean)
    # results_acc_median = np.array(results_acc_median)
    # results_comp_mean = np.array(results_comp_mean)
    # results_comp_median = np.array(results_comp_median)
    
    # compute stat of results
    # mean_acc = np.mean(results_acc_mean)
    # median_acc = np.mean(results_acc_median)
    # mean_comp = np.mean(results_comp_mean)
    # median_comp = np.mean(results_comp_median)
    # f.write("mean acc: " + str(mean_acc) + '\n')
    # f.write("median acc: " + str(median_acc) + '\n')
    # f.write("mean comp: " + str(mean_comp) + '\n')
    # f.write("median comp: " + str(median_comp) + '\n')

# method 3, single process
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
