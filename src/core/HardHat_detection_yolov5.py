import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from core.models.experimental import attempt_load
from core.utils.datasets import LoadStreams, LoadImages
from core.utils.general import non_max_suppression, scale_coords, xyxy2xywh, plot_one_box, letterbox
from core.utils.torch_utils import select_device, time_synchronized
import numpy as np
import matplotlib.pyplot as plt


class Hardhat_detection_yolov5():
    def __init__(self, view, draw):
        if torch.cuda.is_available():
            device_type = 'cuda:0'
        else:
            device_type = 'cpu'

        print('Using device:', device_type)
        self.device = select_device(device_type)
        self.view = view
        self.draw = draw

        # Load self.model
        weights = '../../models/best.pt'
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 self.model
        
        # Get self.names and self.colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # Set constant variables
        self.imgsz = 640

        self.conf_thres = 0.4
        self.iou_thres = 0.5
        self.classes = None
        self.agnostic_nms = True
        self.colors = [[0,0,255], [255,255,255], [255, 255, 0], [255,0,0], [210,105,30]]
        # names: ['blue', 'white', 'yellow', 'red', 'none']


    def predict(self, img_ori):
        # Run inference
    
        img = letterbox(img_ori, new_shape=self.imgsz)[0]
    
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
    
        img = torch.from_numpy(img).to(self.device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
    
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = self.model(img, augment=True)[0]

        t2 = time_synchronized()
        print(f'Model inference FPS: {1 / (t2 - t1)}')
    
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        bboxes_xyx2y2 = []
        labels = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # gn = torch.tensor(img_ori.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_ori.shape).round()
    
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    bboxes_xyx2y2.append([int(coord) for coord in xyxy])
                    labels.append(self.names[int(cls)])

                    if self.draw:   # Add bbox to image
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        plot_one_box(xyxy, img_ori, label=label, color=self.colors[int(cls)], line_thickness=3)

                # Stream results
                if self.view:
                    plt.figure(figsize=(15, 15))
                    plt.imshow(img_ori[:, :, [2, 1, 0]])
                    plt.show()

                return bboxes_xyx2y2, labels, img_ori

        return bboxes_xyx2y2, labels, img_ori


# hardhat_detector = Hardhat_detection_yolov5()
#
# source_img_path = '../data/00000.jpg'
# img_ori = cv2.imread(source_img_path)
# with torch.no_grad():
#     t1 = time_synchronized()
#     det, gn = hardhat_detector.predict(img_ori)
#     t2 = time_synchronized()
#     print(f'FPS: {round((t2 - t1), 3)}')
#
