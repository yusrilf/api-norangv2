import argparse
import csv
import os
from pathlib import Path
import torch
from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (increment_path, scale_boxes, non_max_suppression, check_img_size)
from utils.torch_utils import select_device
from utils.torch_utils import smart_inference_mode

@smart_inference_mode()
def detect_vin(weights='best.pt', source='data/images', imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45, 
               device='cpu', save_txt=False, nosave=True, classes=None, agnostic_nms=False, project='runs/detect', 
               name='exp', exist_ok=False, vid_stride=1):

    source = str(source)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make directory

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=None)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    vin_sequence = ""  # Store predicted VIN sequence here

    for path, im, im0s, _, _ in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.float() / 255.0  # scale to [0, 1]
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dimension

        # Inference
        pred = model(im)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)

        for det in pred:  # per image
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                sorted_detections = sorted(det, key=lambda x: x[0])  # Sort left to right

                for *xyxy, conf, cls in sorted_detections:
                    c = int(cls)
                    label = names[c]
                    vin_sequence += label  # Append predicted character to VIN sequence

    return vin_sequence
