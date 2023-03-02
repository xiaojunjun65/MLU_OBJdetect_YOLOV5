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
from utils.datasets import LoadImagesByTxt
from utils.general import (Logging, check_img_size, colorstr, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync

os.environ['TORCH_MIN_CNLOG_LEVEL'] = '3'

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
def run_transplant(opt):

    nc = opt.num_classes
    
    if opt.fuse:
        orig_model = Model(opt.cfg, ch=3, nc=nc).fuse().eval()
    else:
        orig_model = Model(opt.cfg, ch=3, nc=nc).eval()
    orig_model.change_tmpshape(opt.imgsz, 1)

    if opt.device == 'mlu':
        model = mlu_quantize.adaptive_quantize(orig_model, steps_per_epoch=1, bitwidth=16, inplace=True)
    else:
        model = orig_model
    
    model.load_state_dict(torch.load(opt.model)['model'].float().state_dict(), strict=False)
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
    ct.set_device(-1)

    os.makedirs(os.path.abspath(os.path.dirname(opt.offline_model)), exist_ok=True)
    ct.save_as_cambricon(opt.offline_model.split('.cambricon')[0])

    randn_input = torch.randn(opt.batch_size, 3, *imgsz).float()
    fuse_model = torch.jit.trace(quantized_model.to(ct.mlu_device()), randn_input.to(ct.mlu_device()), check_trace = False)

    with torch.no_grad():
        fuse_model(randn_input.to(ct.mlu_device()))
        
    ct.save_as_cambricon("")

    shutil.rmtree(opt.quantized_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model.pt model.pth path(s)')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max_det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--fuse', action='store_true', help='fuse model')

    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--num_classes", type=int, required=True, help="num classes of model")
    parser.add_argument('--image_number', type=int, default=1, help='test image number')
    parser.add_argument("--quantized_mode", type=int, default=1, choices=[0, 1], help ="0-int8 1-int16")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of offline model')
    parser.add_argument("--core_number", type=int, default=16, help="Core number of offline model with simple compilation")
    parser.add_argument("--core_version", type=str, default="MLU270", help="Specify the offline model run device type")
    parser.add_argument('--offline_model', type=str, default=None, help='save offline model path')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    print(opt)

    log_dir = os.path.dirname(opt.model)
    os.makedirs(log_dir, exist_ok=True)
    LOGGER = Logging(log_dir+os.sep+'transplant_log.txt').logger
    try:
        run_transplant(opt)
    except Exception as e:
        LOGGER.info(traceback.format_exc())