import cv2 as cv
import numpy as np

from .image_to_megapix_scaler import ImageToMegapixScaler

class Downscaler(ImageToMegapixScaler):

    @staticmethod
    def force_downscale(scale):
        return min(1.0, scale)

    def get_scale_by_resolution(self, resolution):
        return self.force_downscale(
            super().get_scale_by_resolution(resolution)
            )

    def get_scale_by_image(self, img):
        return self.force_downscale(
            super().get_scale_by_image(img)
            )
