import os
import operator
import numpy as np
import pandas as pd

#================================Create bounding box object================================
class BBox:
    def __init__(self, x1, x2, y1, y2, conf_score, label, candidate_lst, score_cls_lst):
        self.x1 = x1       
        self.y1 = y1     
        self.x2 = x2        
        self.y2 = y2      
        self.conf_score = conf_score
        self.label = label
        self.candidate_lst = candidate_lst
        self.score_cls_lst = score_cls_lst
        
        self.x_center = (self.x1+self.x2)/2
        self.y_center = (self.y1+self.y2)/2
        self.CXCYDifference = self.x_center - self.y_center

def createListOfBBoxes(img_df):
    listOfBBoxes = []
    for _, row in img_df.iterrows():
        x1 = float(row["x1"]) 
        y1 = float(row["y1"]) 
        x2 = float(row["x2"]) 
        y2 = float(row["y2"]) 
        conf_score = float(row["conf"])
        label = row["label"]
        candidate_lst = row["candidate"]
        score_cls_lst = row["score_cls"]
        
        # Create object
        box = BBox(x1, x2, y1, y2, conf_score, label, candidate_lst, score_cls_lst)
        listOfBBoxes.append(box)

    return listOfBBoxes

#================================Post processing================================
#================================SORT================================
# Check if bounding box belongs to the column identified by the anchor
def checkConverage(bbox, anchor):
    return anchor[0]<= bbox.x_center and bbox.x_center <= anchor[1]

def SortBBoxRTLAndTTB(listOfBBoxes):

    # Get the top-most and right-most bounding box
    fistBBox=None
    index=None
    maxCXCYDifference=-10000
    for i in range(0, len(listOfBBoxes)):
        bbox = listOfBBoxes[i]
        if bbox.CXCYDifference > maxCXCYDifference:
            fistBBox = bbox
            index=i
            maxCXCYDifference = bbox.CXCYDifference
    listOfBBoxes.pop(index)

    # Add bounding boxes to the column they belong
    columns = [[fistBBox]]
    anchors = [[fistBBox.x1, fistBBox.x2]]
    listOfBBoxes.sort(key=operator.attrgetter('x_center'), reverse = True)
    for bbox in listOfBBoxes:
        toAddNewColumn = True
        for i in range(0, len(columns)):
            if toAddNewColumn == False: 
                break
            cur_anchor = anchors[i]
            isInCurrentColumn = checkConverage(bbox=bbox, anchor=cur_anchor)
            if isInCurrentColumn == True:
                columns[i].append(bbox)
                toAddNewColumn = False
        if toAddNewColumn == True:
            columns.append([bbox])
            anchors.append([bbox.x1, bbox.x2])

    # Sort columns descending by anchor bbox
    anchors, columns = zip(*sorted(zip(anchors, columns), key=lambda x: x[0], reverse = True))

    # Sort each column members ascending by CY
    for i in range(0, len(columns)):
        columns[i].sort(key=operator.attrgetter('y_center'))
    
    # Flatten columns array
    listOfSortedBBoxes = [bbox for column in columns for bbox in column]
    return listOfSortedBBoxes

########################################################################################################################
#================================SPLIT LINES================================
def CalculateBBoxCenterAndSize(bbox_lst):
    left = bbox_lst[0]
    top = bbox_lst[1]
    right = bbox_lst[2]
    bottom = bbox_lst[3]
    center_x = bbox_lst[4]
    center_y = bbox_lst[5]
    
    width = right - left
    height = bottom - top
    xywh = np.array([center_x, center_y, width, height])

    return xywh


def CompareCenterAndSize(xywh_1, xywh_2, hyper_lambda=1.5):
    if (abs(xywh_1[0] - xywh_2[0]) > hyper_lambda*((xywh_1[2] + xywh_2[2])/2)) \
    or (abs(xywh_1[1] - xywh_2[1]) > hyper_lambda*((xywh_1[3] + xywh_2[3])/2)):
        return True
    return False


def SplitLine(bbox_lst, hyper_lambda=1.5):
    sep_lst = []
    for idx in range(len(bbox_lst)-1):
        xywh_1 = CalculateBBoxCenterAndSize(bbox_lst[idx])
        xywh_2 = CalculateBBoxCenterAndSize(bbox_lst[idx+1])

        if CompareCenterAndSize(xywh_1, xywh_2, hyper_lambda):
            sep_lst.append("\n")
        else:
            sep_lst.append(" ")
    return sep_lst


                
