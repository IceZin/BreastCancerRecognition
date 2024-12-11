import cv2
import numpy as np


class ImageNotFound(Exception):
    pass


class XRayImage:
    def __init__(self, path):
        self.image = cv2.imread(path)

        if self.image is None:
            raise ImageNotFound("Image not found")

        self.path = path
        self.processed_image = self.image.copy()
        self.resize_scale = [1, 1]
        self.attributes = {}

        self.apply_contrast(1.5, 0)

    def apply_contrast(self, gain, bias):
        for y in range(self.image.shape[0]):
            for x in range(self.image.shape[1]):
                self.processed_image[y, x] = np.clip(
                    gain * self.image[y, x] + bias, 0, 255
                )

    def draw_area(self, points):
        mask = self.processed_image.copy()
        cv2.fillPoly(mask, np.array([points]), (255, 0, 0))
        self.processed_image = cv2.addWeighted(mask, 0.5, self.processed_image, 0.5, 0)

    def create_mask(self, mask_type, points):
        mask = np.zeros(self.processed_image.shape)

        if mask_type == "polygon":
            cv2.fillPoly(mask, np.array([points]), (255, 255, 255))
        elif mask_type == "ellipse":
            print(points)
            cv2.ellipse(mask, points[0], points[1], 0.0, 0.0, 360, (255, 255, 255), -1)
        elif mask_type == "circle":
            cv2.circle(mask, points[0], points[1][0], (255, 255, 255), -1)

        return mask

    def resize_point(self, point):
        return (
            round(point[0] / self.resize_scale[0]),
            round(point[1] / self.resize_scale[1]),
        )

    def resize(self, width, height):
        self.resize_scale[0] = self.image.shape[1] / width
        self.resize_scale[1] = self.image.shape[0] / height

        print(self.resize_scale)

        self.processed_image = cv2.resize(self.image, (width, height))
