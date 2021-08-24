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

    def estimate_original_img_size(self, resized_img):
        """ sizes in (width, height) tuples """
        downscaled_img_size = np.array([resized_img.shape[1],
                                        resized_img.shape[0]])
        return (downscaled_img_size / self.scale).astype(int).tolist()

    def get_aspect_to(self, scaler):
        if self.is_scale_set and scaler.is_scale_set:
            return self.scale / scaler.scale
        else:
            print("Scale not set")
            exit()
