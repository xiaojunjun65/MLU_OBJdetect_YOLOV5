import os
import json
from tkinter import Y
import cv2

from matplotlib import image

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

def parse_json_info(json_info, shape_type='rectangle'):
    json_info_list = []
    for item in json_info:
        if item['shape_type'] != shape_type:
            continue
        label = item['label']
        points = item['points']
        xmin, ymin, xmax, ymax = points[0][0], points[0][1], points[1][0], points[1][1]
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
        json_info_list.append([label, 0.5*(xmin+xmax), 0.5*(ymin+ymax), (xmax-xmin+1), (ymax-ymin+1)])
    return json_info_list

def extrotec2yolo(dataset_path, label2num):
    files = os.listdir(dataset_path)
    img_files = [os.path.join(dataset_path, x) for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

    for img_file in img_files:
        img_format = '.' + img_file.split('.')[-1]
        json_file = img_file.replace(img_format, '.json')
        txt_file = img_file.replace(img_format, '.txt')
        if os.path.isfile(json_file):
            with open(json_file, 'r') as fp:
                json_info = json.load(fp)
        else:
            json_info = []
        json_info_list = parse_json_info(json_info)
        if len(json_info) == 0 or len(json_info_list) == 0:
            fp = open(txt_file, 'w')
            fp.close()
        else:
            image = cv2.imread(img_file)
            h, w, _ = image.shape
            label_info_list = [[label2num[item[0]], item[1]/w, item[2]/h, item[3]/w, item[4]/h] for item in json_info_list]
            with open(txt_file, 'w') as fp:
                for item in label_info_list:
                    fp.write('{} {} {} {} {}\n'.format(item[0], item[1], item[2], item[3], item[4]))

def yolo2extrotec(dataset_path, num2label):
    files = os.listdir(dataset_path)
    img_files = [os.path.join(dataset_path, x) for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

    for img_file in img_files:
        img_fromat = '.' + img_file.split('.')[-1]
        txt_file = img_file.replace(img_fromat, '.txt')
        json_file = img_file.replace(img_fromat, '.json')
        if not os.path.isfile(txt_file):
            continue
        with open(txt_file) as fp:
            lines = fp.readlines()
        if len(lines) == 0:
            continue
        json_info = []
        image = cv2.imread(img_file)
        img_h, img_w, _ = image.shape
        for line in lines:
            item = {}
            num, x, y, w, h = line.split(' ')
            num, x, y, w, h = int(num), float(x), float(y), float(w), float(h)
            x1, y1, x2, y2 = (x - 0.5*w)*img_w, (y-0.5*h)*img_h, (x + 0.5*w)*img_w, (y+0.5*h)*img_h
            x1 = 0 if x1 < 0 else x1
            y1 = 0 if y1 < 0 else y1
            x2 = img_w-1 if x2 > img_w-1 else x2
            y2 = img_h-1 if y2 > img_h-1 else y2
            item["label"] = num2label[num]
            item["points"] = [[x1, y1], [x2, y2]]
            item["shape_type"] = "rectangle"
            json_info.append(item)
        with open(json_file, 'w+', encoding='utf8') as fp:
            json.dump(json_info, fp, ensure_ascii=False, indent=2)

def is_extrotec(json_info):
    if not isinstance(json_info, list):
        return False
    elif len(json_info) == 0:
        return True
    elif 'points' in json_info[0].keys():
        return True
    else:
        return False


def is_labelme(json_info):
    if isinstance(json_info, dict):
        return True
    else:
        return False

def is_labelImg(json_info):
    if not isinstance(json_info, list):
        return False
    elif len(json_info) == 0:
        return True
    elif 'annotations' in json_info[0].keys():
        return True
    else:
        return False

def parse_all_json_info(json_info):
    if is_extrotec(json_info):
        json_info_list = []
        for item in json_info:
            if item['shape_type'] != 'rectangle':
                continue
            label = item['label']
            points = item['points']
            xmin, ymin, xmax, ymax = points[0][0], points[0][1], points[1][0], points[1][1]
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
            json_info_list.append([label, 0.5*(xmin+xmax), 0.5*(ymin+ymax), (xmax-xmin+1), (ymax-ymin+1)])
        return json_info_list
    elif is_labelImg(json_info):
        json_info = json_info[0]
        json_info_list = []
        annotations = json_info['annotations']
        for annotation in annotations:
            coordinates = annotation['coordinates']
            label, x, y, width, height = annotation['label'], coordinates['x'], coordinates['y'], coordinates['width'], coordinates['height']
            json_info_list.append([label, float(x), float(y), float(width), float(height)])
        return json_info_list
    elif is_labelme(json_info):
        json_info_list  = []
        return json_info_list
    else:
        return []

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
def all2yolo(dataset_path, label2num):
    if os.path.isfile(dataset_path):
        with open(dataset_path) as fp:
            label_info = fp.readlines()
        dir_name = os.path.dirname(dataset_path)
        img_files = [os.path.join(dir_name, item.strip().split(',')[0]) for item in label_info]
    else:
        files = os.listdir(dataset_path)
        img_files = [os.path.join(dataset_path, x) for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    
    for img_file in img_files:
        img_format = '.' + img_file.split('.')[-1]
        json_file = img_file.replace(img_format, '.json')
        txt_file = img_file.replace(img_format, '.txt')
        if os.path.isfile(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as fp:
                    json_info = json.load(fp)
            except Exception as e:
                json_info = []
        else:
            json_info = []
        json_info_list = parse_all_json_info(json_info)
        if len(json_info) == 0 or len(json_info_list) == 0:
            fp = open(txt_file, 'w')
            fp.close()
        else:
            image = cv2.imread(img_file)
            h, w, _ = image.shape
            label_info_list = [[label2num[item[0]], item[1]/w, item[2]/h, item[3]/w, item[4]/h] for item in json_info_list]
            with open(txt_file, 'w') as fp:
                for item in label_info_list:
                    fp.write('{} {} {} {} {}\n'.format(item[0], item[1], item[2], item[3], item[4]))
    
    return img_files

if __name__ == '__main__':
    # label2num = {'1':0}
    # dataset_path = '../dataset/rectabgle'
    # extrotec2yolo(dataset_path, label2num)

    num2label = {0:'瑕疵'}
    dataset_path = '../dataset/extrotec_mountain_buckle'
    yolo2extrotec(dataset_path, num2label)

    # label2num = {'NG':0}
    # dataset_path = '../dataset/test'
    # all2yolo(dataset_path, label2num)