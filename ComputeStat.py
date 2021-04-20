import os
import numpy as np
import scipy.io as scio

"""
this python file's function is to read the .mat file saved by BaseEvalMain.py, and compute the dist'mean
"""

if __name__ == "__main__":
    resultPath = "/algo/algo/hanjiawei/mvs_result/baseline_traineval_noise/baseline_noise2"
    # mat_test = scio.loadmat('/algo/algo/hanjiawei/mvs_result/baseline_traineval_noise/baseline_noise2/mvsnet_Eval_1.mat')
    # read the mat files in resultPath, return a name_path list
    name_list = os.listdir(resultPath)
    mat_list = []
    for name in name_list:
        if name.endswith(".mat"):
            mat_list.append(os.path.join(resultPath,name))
    print(mat_list)
    # list to save the accurate and complete distance
    acc_list = []
    comp_list = []
    for mat in mat_list:
        mat_data = scio.loadmat(mat)
        ddata = np.mean(mat_data['FilteredDdata'])
        dstl = np.mean(mat_data['FilteredDstl'])
        print("scan" + str(mat_data['cSet'][0]) + " the acc: {:.4f}  the comp: {:.4f}".format(ddata, dstl))
        acc_list.append(ddata)
        comp_list.append(dstl)

    acc_list = np.array(acc_list)
    acc = np.mean(acc_list)
    comp_list = np.array(comp_list)
    comp = np.mean(comp_list)
    print("mean acc: {:.4f}  mean comp: {:.4f}".format(acc, comp))
