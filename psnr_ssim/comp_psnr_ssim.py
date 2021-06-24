import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import multiprocess as mp
from itertools import product
import cv2
from glob import glob

from util_psnr_ssim import calculate_psnr as psnr
from util_psnr_ssim import _ssim as ssim


parser = argparse.ArgumentParser(description="Compute psnr and ssim from noise imgs and imgs generated by net to gt.")
parser.add_argument("--input_dir", help="the direction of the imgs generated by net")
parser.add_argument("--target_dir", help="the direction of the gt imgs")
args = parser.parse_args()
args.input_dir = "/algo/algo/hanjiawei/HINet/results/HINet-DTU/visualization/DTU/"
args.target_dir = "/algo/algo/hanjiawei/dtu_testing_small/"

def subroutine(params):
    # file name and path
    input_dir = params[0]
    target_dir = params[1]
    scan = params[2]
    img_format = params[3]
    psnr_ssim_txt = os.path.join(input_dir, "psnr_ssim_numpy.txt")
    # psnr ssim list in scan
    noise_psnr_list = []
    noise_ssim_list = []
    net_psnr_list = []
    net_ssim_list = []
    img_name_list = [os.path.splitext(image_name.split('/')[-1])[0] for image_name in os.listdir(os.path.join(target_dir, "dtu", scan, "images"))]
    for img_name in img_name_list:
        #NOTE: select the format of images and numpy or tensor method to calculate panr and ssim
        img_name_jpg = img_name + '.jpg'
        img_name = img_name + img_format
        
        #NOTE: Using Tensor and the method1 to calculate psnr and ssim
        # # according to the images generated direction struction modify the read path
        # # img_net = torch.from_numpy(np.array(Image.open(os.path.join(input_dir, scan, "rgb", img_name))) / 255.0)
        # img_net = torch.from_numpy(np.array(Image.open(os.path.join(input_dir, scan, img_name))) / 255.0)
        # img_net = img_net.cuda()
        # img_net = img_net.permute(2, 0, 1)
        # img_noise = torch.from_numpy(np.array(Image.open(os.path.join(target_dir, "dtu_noise", scan, "images", img_name_jpg))) / 255.0)
        # img_noise = img_noise.permute(2, 0, 1)
        # img_noise = img_noise.cuda()
        # img_gt = torch.from_numpy(np.array(Image.open(os.path.join(target_dir, "dtu", scan, "images", img_name_jpg))) / 255.0)
        # img_gt = img_gt.permute(2, 0, 1)
        # img_gt = img_gt.cuda()
        # img_gt_unsqueeze = img_gt.unsqueeze(0)

        # noise_psnr = psnr(img_noise, img_gt)
        # noise_ssim = ssim(img_noise, img_gt)
        # if img_gt.size() != img_net.size():
        #     img_gt_resize = F.interpolate(img_gt_unsqueeze, size=(img_net.size()[1], img_net.size()[2]), mode='area').squeeze(0)
        #     net_psnr = psnr(img_net, img_gt_resize)
        #     net_ssim = ssim(img_net, img_gt_resize)
        # else:
        #     net_psnr = psnr(img_net, img_gt)
        #     net_ssim = ssim(img_net, img_gt)

        #NOTE: Using Numpy and the method2 to calculate psnr and ssim
        img_net = np.array(Image.open(os.path.join(input_dir, scan, img_name)))
        img_noise = np.array(Image.open(os.path.join(target_dir, "dtu_noise", scan, "images", img_name_jpg)))
        img_gt = np.array(Image.open(os.path.join(target_dir, "dtu", scan, "images", img_name_jpg)))

        noise_psnr = psnr(img_noise, img_gt, crop_border=0)
        noise_ssim = ssim(img_noise, img_gt)
        if img_gt.shape != img_net.shape:
            img_gt_resize = cv2.resize(img_gt, dsize=(img_net.shape[1], img_net.shape[0]), interpolation=cv2.INTER_AREA)
            net_psnr = psnr(img_net, img_gt_resize, crop_border=0)
            net_ssim = ssim(img_net, img_gt_resize)
        else:
            net_psnr = psnr(img_net, img_gt, crop_border=0)
            net_ssim = ssim(img_net, img_gt)
        
    
        noise_psnr_list.append(noise_psnr)
        noise_ssim_list.append(noise_ssim)
        net_psnr_list.append(net_psnr)
        net_ssim_list.append(net_ssim)
        del img_net, img_noise, img_gt

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
    psnr_ssim_txt = os.path.join(args.input_dir, "psnr_ssim_numpy.txt")
    img_format = os.path.splitext(glob(args.input_dir + '*/*')[0])[1]
    params_list = product([args.input_dir], [args.target_dir], test_scans, [img_format])
    with mp.get_context("spawn").Pool(11) as pool:
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


    """
    # hardcode to dtu, if want to change, just need to modify "dtu"
    # net_img_dir = os.path.join(args.input_dir)
    # noise_imgs_dir = os.path.join(args.target_dir, "dtu_noise")
    # gt_imgs_dir = os.path.join(args.target_dir, "dtu")
    # psnr ssim list for all scans
    noise_psnr_list_all = []
    noise_ssim_list_all = []
    net_psnr_list_all = []
    net_ssim_list_all = []

    for scan in test_scans:
        # file name and path
        noise_imgs_dir = os.path.join(args.target_dir, "dtu_noise")
        gt_imgs_dir = os.path.join(args.target_dir, "dtu")
        psnr_ssim_txt = os.path.join(args.input_dir, "psnr_ssim.txt")
        # psnr ssim list in scan
        noise_psnr_list = []
        noise_ssim_list = []
        net_psnr_list = []
        net_ssim_list = []
        img_name_list = [image_name.split('/')[-1] for image_name in os.listdir(os.path.join(args.target_dir, "dtu", scan, "images"))]
        for img_name in img_name_list:
            img_net = torch.from_numpy(np.array(Image.open(os.path.join(args.input_dir, scan, "rgb", img_name))) / 255.0)
            img_net = img_net.permute(2, 0, 1)
            img_noise = torch.from_numpy(np.array(Image.open(os.path.join(noise_imgs_dir, scan, "images", img_name))) / 255.0)
            img_noise = img_noise.permute(2, 0, 1)
            img_gt = torch.from_numpy(np.array(Image.open(os.path.join(gt_imgs_dir, scan, "images", img_name))) / 255.0)
            img_gt = img_gt.permute(2, 0, 1)
            img_gt_unsqueeze = img_gt.unsqueeze(0)
            img_gt_resize = F.interpolate(img_gt_unsqueeze, size=(img_net.size()[1], img_net.size()[2]), mode='area').squeeze(0)
            noise_psnr = psnr(img_noise, img_gt)
            noise_ssim = ssim(img_noise, img_gt)
            net_psnr = psnr(img_net, img_gt_resize)
            net_ssim = ssim(img_net, img_gt_resize)
            noise_psnr_list.append(noise_psnr)
            noise_ssim_list.append(noise_ssim)
            net_psnr_list.append(net_psnr)
            net_ssim_list.append(net_ssim)
        noise_psnr_mean = torch.tensor(noise_psnr_list).mean()
        noise_ssim_mean = torch.tensor(noise_ssim_list).mean()
        net_psnr_mean = torch.tensor(net_psnr_list).mean()
        net_ssim_mean = torch.tensor(net_ssim_list).mean()

        f = open(psnr_ssim_txt, 'a')
        f.write(scan + " noise_psnr_mean: {:.4f}, noise_ssim_mean: {:.4f}, net_psnr_mean: {:.4f}, net_ssim_mean: {:.4f}\n"
            .format(noise_psnr_mean, noise_ssim_mean, net_psnr_mean, net_ssim_mean))
        f.close()

        noise_psnr_list_all.append(noise_psnr_mean)
        noise_ssim_list_all.append(noise_ssim_mean)
        net_psnr_list_all.append(net_psnr_mean)
        net_ssim_list_all.append(net_ssim_mean)
        print("Complete " + scan + " computation!")

    noise_psnr_all_mean = torch.tensor(noise_psnr_list_all).mean()
    noise_ssim_all_mean = torch.tensor(noise_ssim_list_all).mean()
    net_psnr_all_mean = torch.tensor(net_psnr_list_all).mean()
    net_ssim_all_mean = torch.tensor(net_ssim_list_all).mean()
    f = open(psnr_ssim_txt, 'a')
    f.write("for all scans, noise_psnr_mean: {:.4f}, noise_ssim_mean: {:.4f}, net_psnr_mean: {:.4f}, net_ssim_mean: {:.4f}"
            .format(noise_psnr_all_mean, noise_ssim_all_mean, net_psnr_all_mean, net_ssim_all_mean))
    f.close()
"""
