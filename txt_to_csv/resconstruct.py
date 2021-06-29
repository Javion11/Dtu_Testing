"""change the results from disorder txt file to order csv file"""
import os
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="from disorder txt result to order csv file")
parser.add_argument("--input", help="the input txt file path")
args = parser.parse_args()
args.input = "/Users/javion/Desktop/Reconstruction/Reconstruct_results"

txtfile_path = args.input
txtfile_name_list = os.listdir(txtfile_path)
txtfile_path_list = [os.path.join(txtfile_path, txtfile_name) for txtfile_name in txtfile_name_list]
for txtfile_full_name in txtfile_path_list:
    f = open(txtfile_full_name, "r")
    results = f.readlines()
    f.close()
    results.sort(key=lambda x:(len(x.split()[0]), x.split()[0]))
    scan = [result.split()[0] for result in results[1:]]

    acc_mean = []
    acc_median = []
    comp_mean = []
    comp_median = []
    for result in results[1:]:
        acc_mean.append(result.split(',')[0].split()[-1])
        acc_median.append(result.split(',')[1].split()[-1])
        comp_mean.append(result.split(',')[2].split()[-1])
        comp_median.append(result.split(',')[3].split()[-1])
    results_np = np.transpose(np.vstack((acc_mean, acc_median, comp_mean, comp_median))).astype(np.float64)
    results_df = pd.DataFrame(results_np, index=scan, columns=['Acc_mean', 'Acc_median', 'Comp_mean', 'Comp_median'])
    results_mean = np.round(np.mean(results_np, axis=0), 4)
    results_df.loc['mean for all scans'] = results_mean
    results_df.to_csv(os.path.splitext(txtfile_full_name)[0] + '.csv')


