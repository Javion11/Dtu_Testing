import os
import argparse
import torch
import numpy as np
from PIL import Image
import multiprocess as mp
from itertools import product

from util_psnr_ssim import calculate_psnr as psnr
from util_psnr_ssim import _ssim as ssim


parser = argparse.ArgumentParser(description="Compute psnr and ssim from noise imgs to clean imgs.")
parser.add_argument("--noise_dir", help="the direction of the noise images")
parser.add_argument("--clean_dir", help="the direction of the clean imgs")
args = parser.parse_args()
args.noise_dir = "/algo/algo/hanjiawei/dtu_training_small/Rectified_noise/"
args.clean_dir = "/algo/algo/hanjiawei/dtu_training_small/Rectified/"

def subroutine(params):
    # file name and path
    input_dir = params[0]
    target_dir = params[1]
    scan = params[2]
    psnr_ssim_txt = os.path.join(input_dir, "psnr_ssim_numpy.txt")
    # psnr ssim list in scan
    noise_psnr_list = []
    noise_ssim_list = []
    img_name_list = [image_name.split('/')[-1]for image_name in os.listdir(os.path.join(target_dir, scan))]
    for img_name in img_name_list:
        #NOTE: Using Numpy and the method2 to calculate psnr and ssim
        img_noise = np.array(Image.open(os.path.join(input_dir, scan, img_name)))
        img_gt = np.array(Image.open(os.path.join(target_dir, scan, img_name)))

        noise_psnr = psnr(img_noise, img_gt, crop_border=0)
        noise_ssim = ssim(img_noise, img_gt)
    
        noise_psnr_list.append(noise_psnr)
        noise_ssim_list.append(noise_ssim)
        del img_noise, img_gt

    noise_psnr_mean = torch.tensor(noise_psnr_list).mean()
    noise_ssim_mean = torch.tensor(noise_ssim_list).mean()

    f = open(psnr_ssim_txt, 'a')
    f.write(scan + " noise_psnr_mean: {:.4f}, noise_ssim_mean: {:.4f}\n"
        .format(noise_psnr_mean, noise_ssim_mean))
    f.close()
    print("Complete " + scan + " computation!")

if __name__ == "__main__":
    scans = [scan_name.split('/')[-1] for scan_name in os.listdir(args.clean_dir)]
    psnr_ssim_txt = os.path.join(args.noise_dir, "psnr_ssim_numpy.txt")
    params_list = product([args.noise_dir], [args.clean_dir], scans)
    with mp.get_context("spawn").Pool(20) as pool:
            pool.map(subroutine, params_list)

    # compute the stat of all scans
    f_read = open(psnr_ssim_txt, "r")
    results_list = f_read.readlines()
    f_read.close()
    # psnr ssim list for all scans
    noise_psnr_list_all = []
    noise_ssim_list_all = []
    for results in results_list:
        single_result_list = results.split(",")
        for i, single_result in enumerate(single_result_list):
            if i == 0 :
                noise_psnr_list_all.append(float(single_result.split(':')[1]))
            elif i == 1:
                noise_ssim_list_all.append(float(single_result.split(':')[1]))
    noise_psnr_all_mean = torch.tensor(noise_psnr_list_all).mean()
    noise_ssim_all_mean = torch.tensor(noise_ssim_list_all).mean()
    f = open(psnr_ssim_txt, 'a')
    f.write("for all scans, noise_psnr_mean: {:.4f}, noise_ssim_mean: {:.4f}"
            .format(noise_psnr_all_mean, noise_ssim_all_mean))
