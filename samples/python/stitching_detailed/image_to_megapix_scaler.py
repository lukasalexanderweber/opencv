import cv2 as cv
import numpy as np


class ImageToMegapixScaler:
    def __init__(self, megapix):
        self.megapix = megapix
        self.is_scale_set = False
        self.scale = None

    def set_scale_if_not_set(self, scale):
        if self.is_scale_set is False:
            self.scale = scale
            self.is_scale_set = True

    def get_scale_by_image(self, img):
        return self.get_scale_by_resolution(self.get_image_resolution(img))

    def get_scale_by_resolution(self, resolution):
        if self.megapix > 0:
            return np.sqrt(self.megapix * 1e6 / resolution)
        else:
            return 1.0

    @staticmethod
    def get_image_resolution(img):
        return img.shape[0] * img.shape[1]

    def resize(self, img):
        if self.is_scale_set:
            return self.resize_to_scale(img, self.scale)
        else:
            print("Scale not set")
            exit()

    @staticmethod
    def resize_to_scale(img, scale):
        if scale != 1.0:
            return cv.resize(src=img, dsize=None,
                             fx=scale, fy=scale,
                             interpolation=cv.INTER_LINEAR_EXACT)
        else:
            return img

    def get_original_img_sizes_after_resize(self, resized_imgs):
        """ sizes in (width, height) tuples """
        downscaled_img_sizes = [np.array([img.shape[1], img.shape[0]])
                                for img in resized_imgs]
        return [size / self.scale for size in downscaled_img_sizes]

    def get_aspect_to(self, scaler):
        if self.is_scale_set and scaler.is_scale_set:
            return self.scale / scaler.scale
        else:
            print("Scale not set")
            exit()
