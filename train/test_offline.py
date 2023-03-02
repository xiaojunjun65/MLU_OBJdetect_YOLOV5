import argparse
import os
import sys
from pathlib import Path
import traceback

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
from utils.general import (Logging, check_img_size, colorstr, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync

# os.environ['TORCH_MIN_CNLOG_LEVEL'] = '3'

def run_quantize(opt, model):
    # quantized_dir = str(ROOT / 'quantized')
    quantized_dir = os.path.join(os.path.dirname(opt.model), 'quantized')
    opt.quantized_dir = quantized_dir
    os.makedirs(quantized_dir, exist_ok=True)

    mean = [0.0, 0.0, 0.0]
    std  = [1.0, 1.0, 1.0]
    qconfig = {'iteration': opt.image_number, 'use_avg':False, 'data_scale':1.0, 'mean': mean, 'std': std, 'per_channel': True, 'firstconv': False}

    quantized_model = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype='int8' if opt.quantized_mode == 0 else 'int16', gen_quant=True)

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(opt.imgsz, s=stride)  # check img_size

    # dataset = LoadImagesByTxt(str(ROOT / 'runs/train.txt'), img_size=imgsz, stride=stride, auto=False)
    dataset = LoadImagesByTxt(os.path.join(os.path.dirname(opt.model), 'train.txt'), img_size=imgsz, stride=stride, auto=False)

    for index, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        if index >= opt.image_number:
            break
        t1 = time_sync()
        im = torch.from_numpy(im)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()

        # Inference
        with torch.no_grad():
            pred = quantized_model(im, augment=opt.augment, visualize=opt.visualize)
        t3 = time_sync()

    model_name = opt.model.split('.')[-2].split('/')[-1]
    checkpoint = quantized_model.state_dict()
    if opt.quantized_mode == 0:
        opt.quantize_model_path = '{}/{}-int8.pth'.format(quantized_dir, model_name)
        torch.save(checkpoint, opt.quantize_model_path)
    else:
        opt.quantize_model_path = '{}/{}-int16.pth'.format(quantized_dir, model_name)
        torch.save(checkpoint, opt.quantize_model_path)


@torch.no_grad()
def run_online_detect(opt):
    os.makedirs(opt.output_path, exist_ok=True)

    device = ct.mlu_device()
    
    nc = len(opt.label_name)
    names = opt.label_name
    
    if opt.fuse:
        orig_model = Model(opt.cfg, ch=3, nc=nc).fuse().eval()
    else:
        orig_model = Model(opt.cfg, ch=3, nc=nc).eval()
    orig_model.change_tmpshape(opt.imgsz, 1)

    if opt.device == 'mlu':
        model = mlu_quantize.adaptive_quantize(orig_model, steps_per_epoch=1, bitwidth=16, inplace=True)
    else:
        model = orig_model
    
    model.load_state_dict(torch.load(opt.model)['model'].float().state_dict())
    if opt.device == 'mlu':
        model = mlu_quantize.dequantize(model)

    run_quantize(opt, model)
    
    orig_model.change_tmpshape(opt.imgsz, opt.batch_size)
    quantized_model = mlu_quantize.quantize_dynamic_mlu(orig_model)
    state_dict = torch.load(opt.quantize_model_path)
    quantized_model.load_state_dict(state_dict, strict=False)

    stride = int(orig_model.stride.max())  # model stride
    imgsz = check_img_size(opt.imgsz, s=stride)  # check img_size

    ct.set_cnml_enabled(True)
    ct.set_core_number(opt.core_number)
    ct.set_core_version(opt.core_version)

    randn_input = torch.randn(opt.batch_size, 3, *imgsz).float()
    fuse_model = torch.jit.trace(quantized_model.to(device), randn_input.to(device), check_trace = False)
    with torch.no_grad():
        fuse_model(randn_input.to(device))

    dataset = LoadImages(opt.dataset_path, img_size=imgsz, stride=stride, auto=False)
    
    dt, seen = [0.0, 0.0, 0.0], 0
    for index, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im)
        im = im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        
        im = im.to(device)
        dt[0] += t2 - t1
        
        # Inference
        with torch.no_grad():
            pred = fuse_model(im)
        t3 = time_sync()
        dt[1] += t3 - t2

        pred = pred.cpu()

        # NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = os.path.join(opt.output_path, p.name)  # im.jpg
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (int(n) > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            
            # Save results (image with detections)
            save_path = os.path.splitext(save_path)[0] + '.jpg'
            cv2.imwrite(save_path, im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *opt.imgsz)}' % t)

    LOGGER.info(f"Results saved to {colorstr('bold', opt.output_path)}")
    shutil.rmtree(opt.quantized_dir)


if __name__ == '__main__':
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

    parser.add_argument("--label_name", nargs="+", required=True, help='category order')
    parser.add_argument("--dataset_path", type=str, required=True, help="data set path")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--image_number', type=int, default=1, help='test image number')
    parser.add_argument("--quantized_mode", type=int, default=1, choices=[0, 1], help ="0-int8 1-int16")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of offline model')
    parser.add_argument("--core_number", type=int, default=16, help="Core number of offline model with simple compilation")
    parser.add_argument("--core_version", type=str, default="MLU270", help="Specify the offline model run device type")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    print(opt)
    
    log_dir = os.path.dirname(opt.model)
    os.makedirs(log_dir, exist_ok=True)
    LOGGER = Logging(log_dir+os.sep+'test_offline_log.txt').logger
    try:
        run_online_detect(opt)
    except Exception as e:
        LOGGER.info(traceback.format_exc())