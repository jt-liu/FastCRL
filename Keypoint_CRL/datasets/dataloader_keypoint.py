import time

if __name__ == "__main__" and (__package__ is None or __package__ == ''):
    import os
    import sys

    project_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.insert(0, project_dir)
    # import utility.sub_region_detector  # noqa: F401
    __package__ = "Keypoint_CRL.datasets"
import os
import cv2
import math

import numpy as np
import torch
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from .data_io import *


def AddSaltPepperNoise(src, rate):
    srcCopy = src.copy()
    height, width = srcCopy.shape[0:2]
    noiseCount = int(rate * height * width / 2)
    # add salt noise
    X = np.random.randint(width, size=(noiseCount,))
    Y = np.random.randint(height, size=(noiseCount,))
    srcCopy[Y, X] = 255
    # add black peper noise
    X = np.random.randint(width, size=(noiseCount,))
    Y = np.random.randint(height, size=(noiseCount,))
    srcCopy[Y, X] = 0
    return srcCopy


def AddGaussNoise(src, sigma):
    mean = 0
    height, width, channels = src.shape[0:3]
    gauss = np.random.normal(mean, sigma, (height, width, channels))
    noisy_img = src + gauss
    # convert to uint8
    cv2.normalize(noisy_img, noisy_img, 0, 255, cv2.NORM_MINMAX)
    return noisy_img


def rotate_point(pt, center, angle):
    radian = math.radians(angle)
    vec = [pt[0] - center[0], pt[1] - center[1]]

    c = math.cos(radian)
    s = math.sin(radian)
    x = vec[0] * c - vec[1] * s
    y = vec[0] * s + vec[1] * c
    vec = [x, y]

    rotated = [vec[0] + center[0], vec[1] + center[1]]
    return rotated


class KeypointDataset(Dataset):
    def __init__(self, datapath, list_filename, training, imgz, radius, scale_factor=4):
        self.datapath = datapath
        self.image_filenames, self.labels = self.load_path(list_filename)
        self.training = training
        self.image_size = imgz  # H, W
        self.radius = radius
        self.scale_factor = scale_factor  # down sample rate

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split('\t') for line in lines]
        images = [x[0] for x in splits]
        labels = [x[1:] for x in splits]
        return images, labels

    def load_image(self, filename):
        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img  # shape: H, W, C

    def GridHeatmap(self, gt, radius):
        c_x, c_y = round(gt[0] / self.scale_factor), round(gt[1] / self.scale_factor)
        y, x = np.ogrid[:self.image_size[0] // self.scale_factor, :self.image_size[1] // self.scale_factor]

        # D2 = (x - c_x) ** 2 + (y - c_y) ** 2
        # E2 = 2.0 * radius * radius
        # heatmap = np.exp(-D2 / E2)
        distances = np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2)
        heatmap = np.exp(-distances / radius)
        # heatmap[heatmap < 0.0001] = 0.0
        heatmap = heatmap.astype(np.float32)
        # nonzero_points = np.column_stack(np.nonzero(heatmap))
        # _, radius = cv2.minEnclosingCircle(nonzero_points)  # thr=0.01 => heatmap radius=9
        return heatmap

    def OffsetMap(self, gt):
        # offset_x = np.zeros((self.image_size[0] // self.scale_factor, self.image_size[1] // self.scale_factor),
        #                     dtype=np.float32)
        # offset_y = np.zeros((self.image_size[0] // self.scale_factor, self.image_size[1] // self.scale_factor),
        #                     dtype=np.float32)  # yH, xW
        offset_x = np.full((self.image_size[0] // self.scale_factor, self.image_size[1] // self.scale_factor), 0.4)
        offset_y = np.full((self.image_size[0] // self.scale_factor, self.image_size[1] // self.scale_factor), 0.4)  # yH, xW
        gt_loc = tuple(int(item // self.scale_factor) for item in gt)  # x, y
        # print(gt, gt_loc)

        for x in range(gt_loc[0] - self.radius // self.scale_factor - 1,
                       gt_loc[0] + self.radius // self.scale_factor + 1):
            for y in range(gt_loc[1] - self.radius // self.scale_factor - 1,
                           gt_loc[1] + self.radius // self.scale_factor + 1):
                if 0 <= x < offset_x.shape[1] and 0 <= y < offset_x.shape[0]:

                    offset_x[y, x] = (gt[0] - x * self.scale_factor) / self.radius
                    offset_y[y, x] = (gt[1] - y * self.scale_factor) / self.radius

        # for x in range(gt_loc[0] - self.radius // self.scale_factor - 1,
        #                gt_loc[0] + self.radius // self.scale_factor + 1):
        #     for y in range(gt_loc[1] - self.radius // self.scale_factor - 1,
        #                    gt_loc[1] + self.radius // self.scale_factor + 1):
        #         if 0 <= x < offset_x.shape[1] and 0 <= y < offset_x.shape[0]:
        #             pred = ((x * self.scale_factor) + (offset_x[y, x] * self.radius),
        #                     (y * self.scale_factor) + (offset_y[y, x] * self.radius))
        #
        #             if gt[0] - pred[0] > 0.0001 or gt[1] - pred[1] > 0.0001:
        #                 print(gt, pred)
        #                 print(x, self.scale_factor, offset_x[y, x], self.radius)
        #                 time.sleep(1)
        # print(offset_x.max(), offset_x.min())
        return offset_x, offset_y

    def get_random_square_area(self, image, center_point, min_size=20, max_size=50):
        height, width = self.image_size
        # 生成随机区域大小
        size = np.random.randint(min_size, max_size + 1)
        # 确保生成的区域在图像范围内
        top = max(center_point[1] - size, 0)
        bottom = min(center_point[1] + size, height - 1)
        left = max(center_point[0] - size, 0)
        right = min(center_point[0] + size, width - 1)

        random_area = image[int(top):int(bottom) + 1, int(left):int(right) + 1, :]

        return random_area

    def paste_random_area(self, image, random_area):
        # 获取图像尺寸
        height, width = self.image_size

        # 生成随机位置
        random_x = int(np.random.randint(0, width - random_area.shape[1] + 1))
        random_y = int(np.random.randint(0, height - random_area.shape[0] + 1))

        # 复制随机区域到图像
        image[random_y:random_y + random_area.shape[0], random_x:random_x + random_area.shape[1], :] = random_area

        return image

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        img = self.load_image(os.path.join(self.datapath, self.image_filenames[index]))
        # if len(self.labels[index]) != 8:
        #     print(self.image_filenames[index], len(self.labels[index]))
        ori_h, ori_w, _ = img.shape
        r = min(self.image_size[0] / ori_h, self.image_size[1] / ori_w)
        random_size = np.random.randint(70, 100) / 100 if random.random() > 0.5 and self.training else 1
        resize_w, resize_h = int(round(ori_w * r * random_size)), int(round(ori_h * r * random_size))
        img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        pad_width = self.image_size[1] - resize_w
        pad_height = self.image_size[0] - resize_h
        dw = pad_width / 2
        dh = pad_height / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # project the coordinates to self.image_size
        points = self.labels[index]
        points = [float(points[0]) * resize_w / ori_w, float(points[1]) * resize_h / ori_h,
                  float(points[2]) * resize_w / ori_w, float(points[3]) * resize_h / ori_h,
                  float(points[4]) * resize_w / ori_w, float(points[5]) * resize_h / ori_h]
        points = [[points[0] + left, points[1] + top],
                  [points[2] + left, points[3] + top],
                  [points[4] + left, points[5] + top]]

        # ===generate the heatmap===
        heatmap_list = []
        for point_i in range(len(points)):
            heatmap = self.GridHeatmap(points[point_i], 9)
            heatmap_list.append(heatmap)
        if self.training:
            # ===numpy image augment===
            # ===random copy and paste===
            for point_i in range(len(points)):
                if random.random() > 0.7:
                    random_area = self.get_random_square_area(img, [points[point_i][0], points[point_i][1]])
                    salt_rate = np.random.randint(1, 6) / 100
                    img = AddSaltPepperNoise(img, salt_rate)
                    kernel_size = np.random.randint(0, 10)
                    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
                    if kernel_size > 1:
                        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
                    if random.random() > 0.5:
                        random_area = cv2.flip(random_area, 1)
                    img = self.paste_random_area(img, random_area)
            # ===random noise===
            if random.random() > 0.6:
                salt_rate = np.random.randint(1, 12) / 100
                img = AddSaltPepperNoise(img, salt_rate)
            # ===random blur===
            if random.random() > 0.5:
                degree = np.random.randint(1, 15)
                angle = 45
                M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
                motion_blur_kernel = np.diag(np.ones(degree))
                motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
                motion_blur_kernel = motion_blur_kernel / degree
                img = cv2.filter2D(img, -1, motion_blur_kernel)
                kernel_size = np.random.randint(0, 10)
                kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
                if kernel_size > 1:
                    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
                cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        img = np.uint8(img)
        label = np.stack(heatmap_list)
        to_tensor = get_transform()
        img = to_tensor(img)
        label = torch.Tensor(label)
        if self.training:
            # ========pytorch augments========
            # color augment
            if random.random() > 0.5:
                brightness = np.random.uniform(0.5, 2.0)
                contrast_factor = np.random.uniform(0.5, 2.0)
                hue = np.random.uniform(-0.5, 0.5)
                saturation = np.random.uniform(0.0, 2.0)
                processed_color_aug = get_transform_color_aug(contrast_factor, brightness, hue, saturation)
                img = processed_color_aug(img)
            # random flip
            horizontal = transforms.RandomHorizontalFlip(1)
            vertical = transforms.RandomVerticalFlip(1)
            if random.random() > 0.5:
                img = horizontal(img)
                label = horizontal(label)
                for point_i in range(len(points)):
                    points[point_i][0] = self.image_size[1] - 1 - points[point_i][0]

            if random.random() > 0.5:
                img = vertical(img)
                label = vertical(label)
                for point_i in range(len(points)):
                    points[point_i][1] = self.image_size[0] - 1 - points[point_i][1]

            if random.random() > 0.5:
                angle = random.randint(-60, 60)
                img = TF.rotate(img, angle)
                label = TF.rotate(label, angle)
                for point_i in range(len(points)):
                    center = [self.image_size[1] // 2, self.image_size[0] // 2]
                    points[point_i] = rotate_point(points[point_i], center, -angle)
            if random.random() > 0.5:
                # print(img.shape)  # C, H, W
                H_mask = self.image_size[0] // 6
                img[:, 0: np.random.randint(1, H_mask), :] = 0
                img[:, np.random.randint(self.image_size[0] - H_mask, self.image_size[0]): self.image_size[0], :] = 0
                W_mask = self.image_size[1] // 8
                img[:, :, 0: np.random.randint(1, W_mask)] = 0
                img[:, :, np.random.randint(self.image_size[1] - W_mask, self.image_size[1]): self.image_size[1]] = 0

        # ===generate the offset map===
        offset_maps_x = []
        offset_maps_y = []
        for point_i in range(len(points)):
            offset_x, offset_y = self.OffsetMap(gt=points[point_i])
            offset_maps_x.append(offset_x)
            offset_maps_y.append(offset_y)
        offset_maps_x = np.stack(offset_maps_x)
        offset_maps_y = np.stack(offset_maps_y)
        offset_maps_x = torch.Tensor(offset_maps_x)
        offset_maps_y = torch.Tensor(offset_maps_y)
        points = torch.Tensor(np.array(points))

        return {"image": img, "label": label, "offset_x": offset_maps_x, "offset_y": offset_maps_y, "points": points, 'im_name': self.image_filenames[index]}
