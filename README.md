# sample
```python
import os
import random
from typing import Any, Callable, Optional, Tuple
import torch
import tarfile
import numpy as np
import urllib.request
from PIL import Image
from torchvision.transforms.transforms import RandomHorizontalFlip
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms as T

import cv2
import glob
import imgaug.augmenters as iaa
import albumentations as A
def save_images(n_image,aug_mask):
    n_img = Image.fromarray(n_image)
    n_img.save(f"/misc/prn/PRNet/aug_anomaly_image.png")
    
    aug_mask = Image.fromarray(aug_mask)
    aug_mask.save(f"/misc/prn/PRNet/aug_anomaly_mask.png")


def copy_paste():
        anomaly_img_path = "/misc/prn/PRNet/notebooks/anomaly_image_1.png"
        normal_img_path = "/misc/prn/PRNet/notebooks/black_image_1.png"
        anomaly_mask_path = "/misc/prn/PRNet/notebooks/random_image_1_mask.png"
        #fg_mask_path = "/misc/prn/PRNet/fg_mask/bottle/000.png"
        fg_mask_path = "/misc/prn/PRNet/notebooks/diagonal_image.png"
        fg_mask_path = "/misc/prn/PRNet/notebooks/shifted_diagonal_image.png"
        #fg_mask_path = "/misc/prn/PRNet/fg_mask/metal_nut/000.png"
        in_fg_region = True
        aug = A.Compose([
                #A.RandomRotate90(),
                #A.Flip(),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=1.0),
                #A.ShiftScaleRotate(shift_limit=1, scale_limit=2, rotate_limit=45, p=1.0),

                ])

        image = cv2.imread(anomaly_img_path)  # anomaly sample
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (900, 900, 3)
        n_image = cv2.imread(normal_img_path)  # normal sample
        n_image = cv2.cvtColor(n_image, cv2.COLOR_BGR2RGB)  # (900, 900, 3)
        img_height, img_width = n_image.shape[0], n_image.shape[1]

        mask = Image.open(anomaly_mask_path)
        mask = np.asarray(mask)  # (900, 900)
        
        # augmente the abnormal region
        augmentated = aug(image=image, mask=mask)
        aug_image, aug_mask = augmentated['image'], augmentated['mask']
        if in_fg_region:
            fg_mask = Image.open(fg_mask_path)
            fg_mask = np.asarray(fg_mask)
            
            intersect_mask = np.logical_and(fg_mask == 255, aug_mask == 255)
            if (np.sum(intersect_mask) > int(2 / 3 * np.sum(aug_mask == 255))):
                # when most part of aug_mask is in the fg_mask region 
                # copy the augmentated anomaly area to the normal image
                n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                
                save_images(n_image,aug_mask)
                return n_image, aug_mask
            else:
                contours, _ = cv2.findContours(aug_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                center_xs, center_ys = [], []
                widths, heights = [], []
                for i in range(len(contours)):
                    M = cv2.moments(contours[i])
                    if M['m00'] == 0:  # error case
                        x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                        y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                        center_x = int((x_min + x_max) / 2)
                        center_y = int((y_min + y_max) / 2)
                    else:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                    center_xs.append(center_x)
                    center_ys.append(center_y)
                    x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                    y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                    width, height = x_max - x_min, y_max - y_min
                    widths.append(width)
                    heights.append(height)
                if len(widths) == 0 or len(heights) == 0:  # no contours
                    n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                    save_images(n_image,aug_mask)
                    return n_image, aug_mask
                else:
                    max_width, max_height = np.max(widths), np.max(heights)
                    center_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    center_mask[int(max_height/2):img_height-int(max_height/2), int(max_width/2):img_width-int(max_width/2)] = 255
                    fg_mask = np.logical_and(fg_mask == 255, center_mask == 255)

                    x_coord = np.arange(0, img_width)
                    y_coord = np.arange(0, img_height)
                    xx, yy = np.meshgrid(x_coord, y_coord)
                    # coordinates of fg region points
                    xx_fg = xx[fg_mask]
                    yy_fg = yy[fg_mask]
                    xx_yy_fg = np.stack([xx_fg, yy_fg], axis=-1)  # (N, 2)
                    
                    if xx_yy_fg.shape[0] == 0:  # no fg
                        n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                        save_images(n_image,aug_mask)
                        return n_image, aug_mask

                    aug_mask_shifted = np.zeros((img_height, img_width), dtype=np.uint8)
                    for i in range(len(contours)):
                        aug_mask_shifted_i = np.zeros((img_height, img_width), dtype=np.uint8)
                        new_aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)
                        # random generate a point in the fg region
                        idx = np.random.choice(np.arange(xx_yy_fg.shape[0]), 1)
                        rand_xy = xx_yy_fg[idx]
                        delta_x, delta_y = center_xs[i] - rand_xy[0, 0], center_ys[i] - rand_xy[0, 1]
                        
                        x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                        y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                        
                        # mask for one anomaly region
                        aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)
                        aug_mask_i[y_min:y_max, x_min:x_max] = 255
                        aug_mask_i = np.logical_and(aug_mask == 255, aug_mask_i == 255)
                        
                        # coordinates of orginal mask points
                        xx_ano, yy_ano = xx[aug_mask_i], yy[aug_mask_i]
                        
                        # shift the original mask into fg region
                        xx_ano_shifted = xx_ano - delta_x
                        yy_ano_shifted = yy_ano - delta_y
                        outer_points_x = np.logical_or(xx_ano_shifted < 0, xx_ano_shifted >= img_width) 
                        outer_points_y = np.logical_or(yy_ano_shifted < 0, yy_ano_shifted >= img_height)
                        outer_points = np.logical_or(outer_points_x, outer_points_y)
                        
                        # keep points in image
                        xx_ano_shifted = xx_ano_shifted[~outer_points]
                        yy_ano_shifted = yy_ano_shifted[~outer_points]
                        aug_mask_shifted_i[yy_ano_shifted, xx_ano_shifted] = 255
                        
                        # original points should be changed
                        xx_ano = xx_ano[~outer_points]
                        yy_ano = yy_ano[~outer_points]
                        new_aug_mask_i[yy_ano, xx_ano] = 255
                        # copy the augmentated anomaly area to the normal image
                        n_image[aug_mask_shifted_i == 255, :] = aug_image[new_aug_mask_i == 255, :]
                        aug_mask_shifted[aug_mask_shifted_i == 255] = 255
                    save_images(n_image,aug_mask_shifted)
                    return n_image, aug_mask_shifted
        else:  # no fg restriction
            # copy the augmentated anomaly area to the normal image
            n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]

            return n_image, aug_mask
    
if __name__ == "__main__":
    copy_paste()
```
