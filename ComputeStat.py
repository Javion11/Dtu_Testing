import os
import argparse
import numpy as np
import scipy.io as scio

"""
this python file's function is to read the .mat file saved by BaseEvalMain.py, and compute the dist'mean, according the MaxDist value.
"""
parser = argparse.ArgumentParser(description="Compute ply distance between the generated and the stl")
parser.add_argument('resultPath', help="the results path")
parser.add_argument('MaxDist', default=20, help='set the dist larger than MaxDist as outliner points')
args = parser.parse_args()
# args.resultPath = 
args.MaxDist = 20

if __name__ == "__main__":
    resultPath = args.resultPath
    MaxDist = args.MaxDist
    # read the mat files in resultPath, return a name_path list
    name_list = os.listdir(resultPath)
    mat_list = []
    for name in name_list:
        if name.endswith(".mat"):
            mat_list.append(os.path.join(resultPath,name))

    # print(mat_list)
    # list to save the accurate and complete distance
    acc_mean_list = []
    acc_median_list = []
    comp_mean_list = []
    comp_median_list = []

    # create the txt file to save results that filter out the distance larger than MaxDist
    results_txt = os.path.join(resultPath, 'results_maxdist_{}.txt'.format(MaxDist))
    if not os.path.exists(results_txt):
            os.system("touch {}".format(results_txt))
        
    for mat in mat_list:
        mat_data = scio.loadmat(mat)
        ddata = mat_data['FilteredDdata']
        dstl = mat_data['FilteredDstl']
        # Filter out dist > MaxDist
        ddata = ddata[ddata < MaxDist]
        dstl = dstl[dstl < MaxDist]

        ddata_mean = np.mean(ddata)
        ddata_median = np.median(ddata)
        dstl_mean = np.mean(dstl)
        dstl_median = np.median(dstl)

        print("scan" + str(mat_data['cSet'][0]) + " the acc: {:.4f}  the comp: {:.4f}".format(ddata_mean, dstl_mean))
        f = open(results_txt, "a")
        f.write("scan" + str(mat_data['cSet'][0]) + " acc_mean: {:.4f}, acc_median: {:.4f}, comp_mean: {:.4f}, comp_median: {:.4f}\n"
                .format(ddata_mean, ddata_median, dstl_mean, dstl_median))
        f.close()
        acc_mean_list.append(ddata_mean)
        acc_median_list.append(ddata_median)
        comp_mean_list.append(dstl_mean)
        comp_median_list.append(dstl_median)



    acc_mean = np.mean(acc_mean_list)
    acc_median = np.mean(acc_median_list)
    comp_mean = np.mean(comp_mean_list)
    comp_median = np.mean(comp_median_list)
    print("mean acc: {:.4f}  mean comp: {:.4f}".format(acc_mean, comp_mean))
    f.open(results_txt, "a")
    f.write("all scans, " + "acc_mean: {:.4f}, acc_median: {:.4f}, comp_mean: {:.4f}, comp_median: {:.4f}"
            .format(acc_mean, acc_median, comp_mean, comp_median))
    f.close()
