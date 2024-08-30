from __future__ import division
import numpy as np
import torchvision.transforms.functional as F


# Random coloring
class RandomContrast(object):
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, sample):
        contrast_factor = self.contrast_factor
        sample = F.adjust_contrast(sample, contrast_factor)
        return sample


class RandomGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, sample):
        gamma = self.gamma
        sample = F.adjust_gamma(sample, gamma)
        return sample


class RandomBrightness(object):
    def __init__(self, brightness):
        self.brightness = brightness

    def __call__(self, sample):
        brightness = self.brightness
        sample = F.adjust_brightness(sample, brightness)
        return sample


class RandomHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, sample):
        hue = self.hue
        sample = F.adjust_hue(sample, hue)
        return sample


class RandomSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, sample):
        saturation = self.saturation
        sample = F.adjust_saturation(sample, saturation)
        return sample
