import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import multiprocess as mp
from itertools import product

from util_psnr_ssim import psnr,ssim,read_pfm


parser = argparse.ArgumentParser(description="Compute psnr and ssim from noise imgs and imgs generated by net to gt.")
parser.add_argument("--input_dir", help="the direction of the imgs generated by net")
parser.add_argument("--target_dir", help="the direction of the gt imgs")
args = parser.parse_args()
args.input_dir = "/algo/algo/hanjiawei/denoise_result/mvsnet_cleanimg/unet_block_conv3x3_nomask/"
args.target_dir = "/algo/algo/hanjiawei/dtu_testing_small/"

def subroutine(params):
    # file name and path
    input_dir = params[0]
    target_dir = params[1]
    scan = params[2]
    psnr_ssim_txt = os.path.join(input_dir, "psnr_ssim_mask.txt")
    # psnr ssim list in scan
    noise_psnr_list = []
    noise_ssim_list = []
    net_psnr_list = []
    net_ssim_list = []
    img_name_list = [image_name.split('/')[-1] for image_name in os.listdir(os.path.join(target_dir, "dtu", scan, "images"))]
    for img_name in img_name_list:
        img_net = torch.from_numpy(np.array(Image.open(os.path.join(input_dir, scan, "rgb", img_name))) / 255.0)
        img_net = img_net.cuda()
        img_net = img_net.permute(2, 0, 1)
        img_noise = torch.from_numpy(np.array(Image.open(os.path.join(target_dir, "dtu_noise", scan, "images", img_name))) / 255.0)
        img_noise = img_noise.permute(2, 0, 1)
        img_noise = img_noise.cuda()
        img_noise_unsqueeze = img_noise.unsqueeze(0)
        img_noise_resize = F.interpolate(img_noise_unsqueeze, size=(img_net.size()[1], img_net.size()[2]), mode='nearest').squeeze(0)
        img_gt = torch.from_numpy(np.array(Image.open(os.path.join(target_dir, "dtu", scan, "images", img_name))) / 255.0)
        img_gt = img_gt.permute(2, 0, 1)
        img_gt = img_gt.cuda()
        img_gt_unsqueeze = img_gt.unsqueeze(0)
        img_gt_resize = F.interpolate(img_gt_unsqueeze, size=(img_net.size()[1], img_net.size()[2]), mode='area').squeeze(0)

        pfm_name = img_name.split('.')[0] + ".pfm"
        confidence = read_pfm(os.path.join(input_dir, scan, "confidence", pfm_name))[0]
        confidence = confidence.copy() # avoid confidence array is not contiguous
        confidence = torch.from_numpy(confidence).to(img_noise.device)
        photo_mask = confidence > 0.8
        del img_noise, img_gt

        noise_psnr = psnr(img_noise_resize * photo_mask, img_gt_resize * photo_mask)
        noise_ssim = ssim(img_noise_resize * photo_mask, img_gt_resize * photo_mask)
        net_psnr = psnr(img_net * photo_mask, img_gt_resize * photo_mask)
        net_ssim = ssim(img_net * photo_mask, img_gt_resize * photo_mask)
        noise_psnr_list.append(noise_psnr)
        noise_ssim_list.append(noise_ssim)
        net_psnr_list.append(net_psnr)
        net_ssim_list.append(net_ssim)
        del img_net, img_noise_resize, img_gt_resize

    noise_psnr_mean = torch.tensor(noise_psnr_list).mean()
    noise_ssim_mean = torch.tensor(noise_ssim_list).mean()
    net_psnr_mean = torch.tensor(net_psnr_list).mean()
    net_ssim_mean = torch.tensor(net_ssim_list).mean()

    f = open(psnr_ssim_txt, 'a')
    f.write(scan + " noise_psnr_mean: {:.4f}, noise_ssim_mean: {:.4f}, net_psnr_mean: {:.4f}, net_ssim_mean: {:.4f}\n"
        .format(noise_psnr_mean, noise_ssim_mean, net_psnr_mean, net_ssim_mean))
    f.close()
    print("Complete " + scan + " computation!")

if __name__ == "__main__":
    test_scans = [scan_name.split('/')[-1] for scan_name in os.listdir(os.path.join(args.target_dir, "dtu"))]
    psnr_ssim_txt = os.path.join(args.input_dir, "psnr_ssim_mask.txt")
    params_list = product([args.input_dir], [args.target_dir], test_scans)
    with mp.get_context("spawn").Pool(8) as pool:
            pool.map(subroutine, params_list)

    # compute the stat of all scans
    f_read = open(psnr_ssim_txt, "r")
    results_list = f_read.readlines()
    f_read.close()
    # psnr ssim list for all scans
    noise_psnr_list_all = []
    noise_ssim_list_all = []
    net_psnr_list_all = []
    net_ssim_list_all = []
    for results in results_list:
        single_result_list = results.split(",")
        for i, single_result in enumerate(single_result_list):
            if i == 0 :
                noise_psnr_list_all.append(float(single_result.split(':')[1]))
            elif i == 1:
                noise_ssim_list_all.append(float(single_result.split(':')[1]))
            elif i == 2:
                net_psnr_list_all.append(float(single_result.split(':')[1]))
            elif i == 3:
                net_ssim_list_all.append(float(single_result.split(':')[1]))
    noise_psnr_all_mean = torch.tensor(noise_psnr_list_all).mean()
    noise_ssim_all_mean = torch.tensor(noise_ssim_list_all).mean()
    net_psnr_all_mean = torch.tensor(net_psnr_list_all).mean()
    net_ssim_all_mean = torch.tensor(net_ssim_list_all).mean()
    f = open(psnr_ssim_txt, 'a')
    f.write("for all scans, noise_psnr_mean: {:.4f}, noise_ssim_mean: {:.4f}, net_psnr_mean: {:.4f}, net_ssim_mean: {:.4f}"
            .format(noise_psnr_all_mean, noise_ssim_all_mean, net_psnr_all_mean, net_ssim_all_mean))