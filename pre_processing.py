import cv2
import numpy as np
import pandas as pd


class BBox:
    def __init__(self, x_center_norm, y_center_norm, width_norm, height_norm, conf_score):
        self.x_center_norm = x_center_norm
        self.y_center_norm = y_center_norm
        self.conf_score = conf_score

        self.x1 = x_center_norm - width_norm/2
        self.y1 = y_center_norm - height_norm/2
        self.x2 = x_center_norm + width_norm/2
        self.y2 = y_center_norm + height_norm/2

        self.CXCYDifference = self.x_center_norm - self.y_center_norm


def createListOfBBoxes(txt_file):
    listOfBBoxes = []
    with open(txt_file, encoding="utf8") as f:
        for line in f:
            row = line.strip()
            row = row.split(" ")

            # Read output of txt file
            x_center_norm = float(row[1])
            y_center_norm = float(row[2])
            width_norm = float(row[3])
            height_norm = float(row[4])
            conf_score = float(row[5])

            # Create object
            box = BBox(x_center_norm, y_center_norm,
                       width_norm, height_norm, conf_score)
            listOfBBoxes.append(box)

    return listOfBBoxes


def create_bbox_df(img_path, txt_path):
    # Read image width and image height
    img_file = img_path
    raw_image = cv2.imread(img_file, 0)
    image_width, image_height = raw_image.shape[1], raw_image.shape[0]

    listOfBBoxes = createListOfBBoxes(txt_file=txt_path)

    bbox_lst = []
    for bbox in listOfBBoxes:
        bbox_lst.append(np.array([int(bbox.x1*image_width), int(bbox.y1*image_height), int(
            bbox.x2*image_width), int(bbox.y2*image_height), bbox.conf_score]))

    bbox_lst = np.array(bbox_lst)

    df = pd.DataFrame({
        "x1": bbox_lst[:, 0],
        "x2": bbox_lst[:, 2],
        "y1": bbox_lst[:, 1],
        "y2": bbox_lst[:, 3],
        "conf": bbox_lst[:, 4],
    })

    return df
