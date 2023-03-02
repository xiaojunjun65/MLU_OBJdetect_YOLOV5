import argparse
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import shutil
import yaml
import cv2
import torch
import torch_mlu.core.mlu_quantize as mlu_quantize
import torch_mlu.core.mlu_model as ct
from models.yolo_transplant import Model
from utils.datasets import LoadImages, LoadImagesByTxt
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.general import (LOGGER, box_iou, check_img_size, colorstr, non_max_suppression, scale_coords, xywhn2xyxy, xyxy2xywhn, xywh2xyxy)
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync
from utils.augmentations import letterbox
import numpy as np
from all2yolo import *

# os.environ['TORCH_MIN_CNLOG_LEVEL'] = '3'
def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

class OnlineTester:
    def __init__(self, ops):
        self.ops = ops
        self.complieAcc = True
        os.makedirs(self.ops.output_path, exist_ok=True)
        
    def iniTransfors(self):
        self.input_size = self.ops.input_size
        if not isinstance(self.input_size, list) and not isinstance(self.input_size, tuple):
            self.input_size = (self.input_size, self.input_size)

    def transforms(self, img, imgsz=640, stride=32):
        img, ratio, pad = letterbox(img, imgsz, stride=stride, auto=False)
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img).float() / 255.0, ratio, pad

    def initModel(self):
        nc = len(self.ops.label_name)
        if self.ops.fuse:
            self.orig_model = Model(self.ops.cfg, ch=3, nc=nc).fuse().eval()
        else:
            self.orig_model = Model(self.ops.cfg, ch=3, nc=nc).eval()
        self.stride = int(self.orig_model.stride.max())
        self.orig_model.change_tmpshape(self.ops.imgsz, 1)
        device = self.ops.device.lower()
        if device == 'mlu':
            self.model = mlu_quantize.adaptive_quantize(self.orig_model, steps_per_epoch=1, bitwidth=16, inplace=True)
        else:
            self.model = self.orig_model
        if self.ops.model in [None, ""] or not os.path.isfile(self.ops.model):
            self.ops.model = self.ops.output_path + os.sep + "best.pth"
        if os.path.isfile(self.ops.model):
            self.model.load_state_dict(torch.load(self.ops.model)['model'].float().state_dict())
        if device == 'mlu':
            self.model = mlu_quantize.dequantize(self.model)
        self.weight_name = self.ops.model.split('/')[-1].split('.')[0]

    def readImage(self):
        assert os.path.exists(self.ops.dataset_path), "The file not exists"
        self.num2label = dict([(idx, label) for idx, label in  enumerate(self.ops.label_name)])
        self.label2num = dict([(label, idx) for idx, label in  enumerate(self.ops.label_name)])
        self.image_list = []
        self.label_list = []
        self.basePath = str(Path(self.ops.dataset_path).parent)
        with open(self.ops.dataset_path) as f:
            allImageFileInfo = f.readlines()
            for info in allImageFileInfo:
                if ',' in info.strip():
                    file, label = info.strip().split(',')
                    self.image_list.append([1, file])
                    self.label_list.append(label)
                else:
                    self.image_list.append([0, info.strip()])
                    self.label_list.append(-1)

    def generateQuantizeModel(self):
        assert self.ops.quantized_dir
        os.makedirs(self.ops.quantized_dir, exist_ok=True)

        if self.ops.quantized_mode == 0:
            quantized_mode = 'int8'
        else:
            quantized_mode = 'int16'

        qconfig = {'iteration': self.ops.iteration, 'use_avg':self.ops.use_avg, 'data_scale':self.ops.data_scale, 'mean': self.ops.mean, 'std': self.ops.std, 'per_channel': self.ops.per_channel, 'firstconv': self.ops.firstconv}
        self.quantize_model = mlu_quantize.quantize_dynamic_mlu(self.model, qconfig_spec=qconfig, dtype=quantized_mode, gen_quant=True)
        for _, image_file in self.image_list[:self.ops.iteration]:
            img = cv2.imread(os.path.join(self.basePath, image_file))
            idata, _, _ = self.transforms(img, self.ops.imgsz, self.stride)
            idata = idata.unsqueeze(0)
            self.quantize_model(idata)
        self.quantize_weight_path = self.ops.quantized_dir+os.sep+f'{self.weight_name}-{quantized_mode}.pth'
        torch.save(self.quantize_model.state_dict(), self.quantize_weight_path)

    def loadQuantizeModel(self):
        self.quantize_model = mlu_quantize.quantize_dynamic_mlu(self.orig_model)
        self.quantize_model.load_state_dict(torch.load(self.quantize_weight_path))

    def testOfflineModel(self):
        
        if self.ops.core_number < 1:
            self.ops.core_number = 1
        if self.ops.core_version == 'MLU220' and self.ops.core_number > 4:
            self.ops.core_number = 4
        if self.ops.core_version == 'MLU270' and self.ops.core_number > 16:
            self.ops.core_number = 16

        if self.ops.draw_result:
            os.makedirs(os.path.join(self.ops.output_path, "images"), exist_ok=True)
        ct.set_cnml_enabled(True)
        ct.set_core_version(self.ops.core_version)
        ct.set_core_number(self.ops.core_number)
        imgsz = check_img_size(self.ops.imgsz, s=self.stride)
        idata = torch.randn(self.ops.batch_size, 3, *imgsz).float()
        self.fuse_model = torch.jit.trace(self.quantize_model.to(ct.mlu_device()), idata.to(ct.mlu_device()), check_trace=False)
        fp = open(os.path.join(self.ops.output_path, 'test_result.txt'), 'w')
        
        iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        jdict, stats, ap, ap_class = [], [], [], []
        single_cls = True if len(self.ops.label_name) <= 1 else False
        nc = len(self.ops.label_name)
        names = {k: v for k, v in enumerate(self.orig_model.names if hasattr(self.orig_model, 'names') else self.orig_model.module.names)}
        for idx, (isLabel, image_file) in enumerate(self.image_list):
            image0 = cv2.imread(os.path.join(self.basePath, image_file))
            image, ratio, pad = self.transforms(image0, self.ops.imgsz, self.stride)
            
            image = image.unsqueeze(0)
            h,w = image.shape[2:]
            h0,w0 = image0.shape[:2]
            if isLabel:
                label_info = self.parse_json(os.path.join(self.basePath, self.label_list[idx]), (w0, h0))
                labels = np.array(label_info)
                if labels.size:
                    labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w0, ratio[1] * h0, padw=pad[0], padh=pad[1])
                nl = len(labels)
                if nl:
                    labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=w, h=h, clip=True, eps=1E-3)
                targets = torch.zeros((nl, 6))
                if nl:
                    targets[:, 1:] = torch.from_numpy(labels)
                targets[:, 2:] *= torch.Tensor([w, h, w, h])
            shapes = [image0.shape, (ratio, pad)]
            
            with torch.no_grad():
                outputs = self.fuse_model(image.to(ct.mlu_device()))
                out = outputs.cpu()    
                out = non_max_suppression(out, self.ops.conf_thres, self.ops.iou_thres, self.ops.classes, self.ops.agnostic_nms, max_det=self.ops.max_det)
                fp.write(image_file)
                for si, pred in enumerate(out):
                    resInfo = ""
                    predn = pred.clone()
                    scale_coords(image[si].shape[1:], predn[:, :4], shapes[0], shapes[1])  # native-space pred
                    if self.ops.draw_result:
                        annotator = Annotator(image0, line_width=3, example=str(self.ops.label_name))
                        images_output_path = f'{os.sep}'.join(self.ops.output_path.split("/")[-2:]) + "/images/" + os.path.split(image_file)[-1]
                        if len(predn) > 0:
                            resInfo += f",{images_output_path},"
                        else:
                            resInfo += f",{-1},"
                    else:
                        resInfo += f",{-1},"
                    for *xyxy, conf, cls in reversed(predn):
                        if self.ops.draw_result:
                            c = int(cls)  # integer class
                            label = f'{self.num2label[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        resInfo += f"{int(xyxy[0])}:{int(xyxy[1])}:{int(xyxy[2])}:{int(xyxy[3])}@{self.num2label[int(cls)]}@{conf:.3f};"
                    if len(predn) <= 0:
                        resInfo += f"{-1}@{-1}@{-1}"
                    if ';' in resInfo:
                        resInfo = resInfo[:-1]
                    fp.write(resInfo+'\n')
                    #Evaluate
                    if isLabel:
                        labels = targets[targets[:, 0] == si, 1:]
                        nl = len(labels)
                        tcls = labels[:, 0].tolist() if nl else []  # target class
                        if len(pred) == 0:
                            if nl:
                                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                            continue

                        if single_cls:
                            pred[:, 5] = 0
                        
                        if nl:
                            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                            scale_coords(image[si].shape[1:], tbox, shapes[0], shapes[1])  # native-space labels
                            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                            correct = process_batch(predn, labelsn, iouv)
                        else:
                            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
                    
                    if self.ops.draw_result:
                        im0 = annotator.result()
                        # Save results (image with detections)
                        save_path = self.ops.output_path + os.sep + "images" + os.sep + os.path.split(image_file)[-1]
                        cv2.imwrite(save_path, im0)
        fp.close()
        count = len(stats)
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        map = 0
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)
            mp, mr, map50, map = 0, 0, 0, 0
        info = {"info":{"accuracy":map50, "label_num": count, "map": map, "mp": mp, "mr": mr}}
        with open(os.path.join(self.ops.output_path, "test_accuracy.json"), "w") as f:
            json.dump(info, f)

    def parse_json(self, json_file, img0sz):
        if not os.path.isfile(json_file):
            return []
        with open(json_file, 'r') as fp:
            json_info = json.load(fp)
        json_info_list = []
        for item in json_info:
            label = item['label']
            points = item['points']
            xmin, ymin, xmax, ymax = points[0][0], points[0][1], points[1][0], points[1][1]
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
            json_info_list.append([self.label2num[label], 0.5*(xmin+xmax)/img0sz[0], 0.5*(ymin+ymax)/img0sz[1], (xmax-xmin+1)/img0sz[0], (ymax-ymin+1)/img0sz[1]])
        return json_info_list


    def worker(self):
        self.initModel()
        self.readImage()
        self.generateQuantizeModel()
        self.loadQuantizeModel()
        self.testOfflineModel()
        shutil.rmtree(self.ops.quantized_dir)



if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ("true", "t", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max_det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--output_path', default='./output', type=str, help='Save output image path')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--fuse', action='store_true', help='fuse model')

    parser.add_argument('--quantized_dir', type=str, default='/workspace/quantized', help='量化模型保存地址')
    parser.add_argument("--label_name", nargs="+", required=True, help='category order')
    parser.add_argument("--dataset_path", type=str, required=True, help="data set path")
    parser.add_argument('--device', default='mlu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--iteration', type=int, default=1, help='test image number')
    parser.add_argument("--quantized_mode", type=int, default=1, choices=[0, 1], help ="0-int8 1-int16")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of offline model')
    parser.add_argument("--core_number", type=int, default=16, help="Core number of offline model with simple compilation")
    parser.add_argument("--core_version", type=str, default="MLU270", help="Specify the offline model run device type")
    parser.add_argument('--mean', nargs='+', default=[0,0,0], help='firstconv使用')
    parser.add_argument('--std', nargs='+', default=[1,1,1], help='firstconv使用')
    parser.add_argument('--per_channel', type=str2bool, default='false', help='通道量化')
    parser.add_argument('--firstconv', type=str2bool, default='false', help='使用firstconv')
    parser.add_argument("--use_avg", type=str2bool, default='false', help ="是否使用均值")
    parser.add_argument('--data_scale', type=float, default=1.0, help='图片缩放')
    parser.add_argument('--draw_result', type=str2bool, default='true', help='绘制')
    ops = parser.parse_args()
    ops.imgsz *= 2 if len(ops.imgsz) == 1 else 1  # expand

    onlineTester = OnlineTester(ops)
    onlineTester.worker()