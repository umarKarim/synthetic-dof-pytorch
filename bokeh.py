import cv2
import numpy as np 
from math import pi as PI
import os
from options import Options
import matplotlib.pyplot as plt 



class Bokeh():
    def __init__(self, opts):
        self.output_dir = opts.output_dir 
        self.depth_blur_size = opts.depth_blur_size       
        self.depth_steps = opts.depth_steps
        self.resolution = opts.resolution

    def __call__(self, im, depth, roi):
        self.roi_depth = roi
        if self.resolution is not None:
            depth = cv2.resize(depth, self.resolution)
            im = cv2.resize(im, self.resolution)
        
        h, w, _ = im.shape
        # changing the blur kernel as per image dimensions. Assume the init kernel size for 640 im
        self.depth_blur_size = int(1.0 * w / 640.0 * self.depth_blur_size)
        depth_res = cv2.resize(depth, (w, h))
        fg_mask = (depth >= self.roi_depth[0]) * (depth <= self.roi_depth[1])
        fg_mask = np.expand_dims(fg_mask, axis=-1)
        fg_mask = np.concatenate((fg_mask, fg_mask, fg_mask), axis=-1)
        fg_crop = im * fg_mask
        bokeh = self.depth_wise_blur(im, depth_res, fg_crop)
        bokeh[fg_mask] = im[fg_mask]
        return bokeh 

    def depth_wise_blur(self, im_orig, depth, crop):
        blur_levels = np.linspace(0, self.depth_blur_size, self.depth_steps)
        blur_levels = [int(x) for x in blur_levels]
        depth_crop = (crop[:, :, 0] > 0.0) * depth 

        # based on code from project-defude 
        depth_size = (depth.max() - depth.min()) / self.depth_steps
        depth_levels = np.arange(depth.min(), depth.max(), depth_size)
        pof = (self.roi_depth.max() + self.roi_depth.min()) / 2.0 
        norm_depth = np.abs(depth - pof)
        norm_depth = norm_depth / norm_depth.max()
        norm_depth = np.expand_dims(norm_depth, -1)
        norm_depth = np.concatenate((norm_depth, norm_depth, norm_depth), -1)

        im = im_orig
        h, w = im.shape[:2] 
        blur_stack = np.zeros((h, w, 3, self.depth_steps))
        for b_count, b in enumerate(blur_levels):
            # getting the disc kernel
            kernel = self.get_disc_kernel(b)
            if kernel is None:
                blur_stack[:, :, :, b_count] = im 
            else:
                blur_stack[:, :, :, b_count] = cv2.filter2D(im, -1, kernel)
        max_blur_im = np.expand_dims(blur_stack[:, :, :, -1], -1)
        blur_stack = np.concatenate((blur_stack, max_blur_im), -1)
        depth_levels = np.concatenate((depth_levels, [1.0]))

        bokeh = np.zeros((h, w, 3))
        for d, d_min in enumerate(depth_levels):
            if d == len(depth_levels) - 1:
                mask = (norm_depth == norm_depth.max())
                blur_im = blur_stack[:, :, :, d]
                temp = blur_im.copy()
                temp[mask == 0] = 0
                bokeh += temp
            else:
                d_max = depth_levels[d + 1]
                mask = (norm_depth >= d_min) * (norm_depth < d_max)
                blur_im = blur_stack[:, :, :, d]
                temp = blur_im.copy() 
                temp[mask == 0] = 0
                bokeh += temp
        return bokeh

    def get_disc_kernel(self, k):
        radius = k // 2
        x = np.linspace(0, k - 1, k)
        xv, yv = np.meshgrid(x, x)
        mask = (xv - radius) * (xv - radius) + (yv - radius) * (yv - radius) < radius * radius 
        mask = 1.0 * mask
        sum_vals = np.sum(mask)
        if sum_vals == 0:
            return None 
        else:
            return mask / sum_vals 


    
