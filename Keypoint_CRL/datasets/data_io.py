import re
import numpy as np
import torchvision.transforms as transforms
from .transforms import RandomContrast, RandomBrightness, RandomHue, RandomSaturation


def get_transform():
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # mean = [0.206, 0.197, 0.193]
    # std = [0.244, 0.235, 0.233]
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()])


def get_transform_flip_aug(horizontal, vertical):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(horizontal),
        transforms.RandomVerticalFlip(vertical)])


def get_transform_color_aug(contrast_factor, brightness, hue, saturation):
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # mean = [0.206, 0.197, 0.193]
    # std = [0.244, 0.235, 0.233]
    return transforms.Compose([
        transforms.ToPILImage(),
        RandomContrast(contrast_factor),
        RandomBrightness(brightness),
        RandomHue(hue),
        RandomSaturation(saturation),
        transforms.ToTensor()])


def read_all_lines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:
        endian = '<'
        scale = -scale
    else:
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
