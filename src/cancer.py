import os
import pydicom
import pandas
import numpy as np
import cv2
import json

from image import ImageNotFound, XRayImage

dataset_path = "./dataset"
output_dataset_path = "./processed_dataset"
train_data = pandas.read_csv("./src/train_data.csv")
xray = None
mask_amount = 0

for i, row in train_data.iterrows():
    image_name = row["filename"]
    image_path = f"{dataset_path}/{image_name}"
    image_dir = f"{output_dataset_path}/{image_name.split(".")[0]}"

    xray_data = json.loads(row["region_shape_attributes"])

    if mask_amount == 0:
        try:
            xray = XRayImage(image_path)
        except ImageNotFound as err:
            continue

        xray.resize(512, 768)

        os.mkdir(image_dir)
        cv2.imwrite(f"{image_dir}/image.jpg", xray.processed_image)

        mask_amount = row["region_count"]

    mask_id = row["region_id"]
    mask_type = xray_data["name"]
    points = []

    if mask_type == "polygon":
        for z, x in enumerate(xray_data["all_points_x"]):
            points.append(xray.resize_point((x, xray_data["all_points_y"][z])))
    elif mask_type == "ellipse":
        points.append(xray.resize_point((xray_data["cx"], xray_data["cy"])))
        points.append(xray.resize_point((xray_data["rx"], xray_data["ry"])))
    elif mask_type == "circle":
        points.append(xray.resize_point((xray_data["cx"], xray_data["cy"])))
        points.append(xray.resize_point((xray_data["r"] or 1, 0)))

    mask = xray.create_mask(mask_type, points)

    cv2.imwrite(f"{image_dir}/mask_{str(mask_id).zfill(3)}.jpg", mask)

    if mask_id + 1 == mask_amount:
        mask_amount = 0
