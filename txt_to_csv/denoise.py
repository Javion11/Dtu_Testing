"""change the results from disorder txt file to order csv file"""
import os
import numpy as np
import pandas as pd
import argparse
from glob import glob


parser = argparse.ArgumentParser(description="from disorder txt result to order csv file")
parser.add_argument("--input", help="the input txt file path")
args = parser.parse_args()
args.input = "/Users/javion/Desktop/Reconstruction/Denoise_results"

txtfile_path = args.input
txtfile_path_list = glob(txtfile_path + '/*/*.txt')
for txtfile_full_name in txtfile_path_list:
    f = open(txtfile_full_name, "r")
    results = f.readlines()
    f.close()
    results.sort(key=lambda x:(len(x.split()[0]), x.split()[0]))
    scan = [result.split()[0] for result in results[1:]]

    noise_psnr = []
    noise_ssim = []
    net_psnr = []
    net_ssim = []
    for result in results[1:]:
        noise_psnr.append(result.split(',')[0].split()[-1])
        noise_ssim.append(result.split(',')[1].split()[-1])
        net_psnr.append(result.split(',')[2].split()[-1])
        net_ssim.append(result.split(',')[3].split()[-1])
    results_np = np.transpose(np.vstack((noise_psnr, noise_ssim, net_psnr, net_ssim))).astype(np.float64)
    results_df = pd.DataFrame(results_np, index=scan, columns=['Noise_PSNR', 'Noise_SSIM', 'Net_PSNR', 'Net_SSIM'])
    results_mean = np.round(np.mean(results_np, axis=0), 4)
    results_df.loc['mean for all scans'] = results_mean
    results_df.to_csv(os.path.splitext(txtfile_full_name)[0] + '.csv')


