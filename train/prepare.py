import os
import random
import string
import shutil
import json
from PIL.Image import new
import cv2
import argparse


def copy_file(origin_dir, ratio=0.8):
    labels_dir = os.path.join(origin_dir, 'label')
    labels = os.listdir(labels_dir)
    for label in labels:
        label_dir = os.path.join(labels_dir, label)
        label_files = os.listdir(label_dir)
        train_label_files = random.sample(label_files, int(len(label_files)*ratio))
        val_label_files = list(set(label_files) - set(train_label_files))
        for label_file in train_label_files:
            label_path = os.path.join(label_dir, label_file)
            image_path = label_path.replace('label', 'image').replace(label_suffix, image_suffix)
            new_name = ''.join(random.sample(string.ascii_letters + string.digits, 16))
            shutil.copy(label_path, os.path.join(train_dir, f'{new_name}.{label_suffix}'))
            shutil.copy(image_path, os.path.join(train_dir, f'{new_name}.{image_suffix}'))
        
        for label_file in val_label_files:
            label_path = os.path.join(label_dir, label_file)
            image_path = label_path.replace('label', 'image').replace(label_suffix, image_suffix)
            new_name = ''.join(random.sample(string.ascii_letters + string.digits, 16))
            shutil.copy(label_path, os.path.join(val_dir, f'{new_name}.{label_suffix}'))
            shutil.copy(image_path, os.path.join(val_dir, f'{new_name}.{image_suffix}'))

def parse_json_info(json_info):
    json_info_list = []
    annotations = json_info['annotations']
    for annotation in annotations:
        coordinates = annotation['coordinates']
        label, x, y, width, height = annotation['label'], coordinates['x'], coordinates['y'], coordinates['width'], coordinates['height']
        json_info_list.append([label, float(x), float(y), float(width), float(height)])
    return json_info_list

def labelImg2yolo(image_files, json_files, delete=True):
    for image_file, json_file in zip(image_files, json_files):
        print(image_file)
        image = cv2.imread(image_file)
        h, w, _ = image.shape
        with open(json_file, 'r') as fp:
            json_info = json.load(fp)
        json_info_list = parse_json_info(json_info[0])
        label_info_list = [[label2num[item[0]], item[1]/w, item[2]/h, item[3]/w, item[4]/h] for item in json_info_list]
        txt_file = json_file.replace('json', 'txt')
        with open(txt_file, 'w') as fp:
            for item in label_info_list:
                fp.write('{} {} {} {} {}\n'.format(item[0], item[1], item[2], item[3], item[4]))
        if delete:
            os.remove(json_file)

def create_empty_txt(origin_dir):
    images_dir = os.path.join(origin_dir, 'image')
    labels = os.listdir(images_dir)
    for label in labels:
        image_dir = os.path.join(images_dir, label)
        image_files = os.listdir(image_dir)
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            json_path = image_path.replace('image', 'label').replace(image_suffix, 'json')
            if not os.path.isfile(json_path):
                new_name = ''.join(random.sample(string.ascii_letters + string.digits, 16))
                shutil.copy(image_path, os.path.join(train_dir, f'{new_name}.{image_suffix}'))
                fp = open(os.path.join(train_dir, f'{new_name}.txt'), 'w')
                fp.close()

def create_reinforce():
    files = os.listdir(reinforce_dir)
    image_files = [file for file in files if os.path.splitext(file) == image_suffix]
    for image_file in image_files:
        image_path = os.path.join(reinforce_dir, image_file)
        label_path = image_path.replace(image_suffix, 'txt')
        if os.path.isfile(image_path) and os.path.isfile(label_path):
            shutil.copy(image_path, os.path.join(target_dir, 'train', image_file))
            shutil.copy(label_path, os.path.join(target_dir, 'train', image_file.replace(image_suffix, 'txt')))

def is_empty_json(json_file):
    with open(json_file, 'r') as fp:
            json_info = json.load(fp)
    json_info_list = parse_json_info(json_info[0])
    if len(json_info_list) == 0:
        return True
    else:
        return False

def copy_ok_data(origin_dir):
    images_dir = os.path.join(origin_dir, 'image')

    labels = os.listdir(images_dir)
    for label in labels:
        image_dir = os.path.join(images_dir, label)
        image_files = os.listdir(image_dir)
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            json_path = image_path.replace('image', 'label').replace(image_suffix, 'json')
            if not os.path.isfile(json_path) or is_empty_json(json_path):
                new_name = ''.join(random.sample(string.ascii_letters + string.digits, 16))
                shutil.copy(image_path, os.path.join(ok_dir, f'{new_name}.{image_suffix}'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=int, help='0:磁环 1:纸巾')
    args = parser.parse_args()

    if args.project == 0:
        label2num = {'裂纹':0, '裂痕':0, '附着':1, '划伤':2, '缺口':3, '沙眼':4, '砂眼':4, '脏污':5, '脏物':5}

        label_suffix = 'json'
        image_suffix = 'tif'

        target_dir = '/home/cjkai/workspace/dataset/MagnetRing'

        origin_dirs = [
            '/home/cjkai/workspace/dataset/MagnetRing/磁环_2022-01-10',
            '/home/cjkai/workspace/dataset/MagnetRing/磁环_2022-01-17',
            '/home/cjkai/workspace/dataset/MagnetRing/磁环_2022-01-19',
            '/home/cjkai/workspace/dataset/MagnetRing/良品图片_2022-01-24',
            '/home/cjkai/workspace/dataset/MagnetRing/不良判为良品_2022-02-18',
            '/home/cjkai/workspace/dataset/MagnetRing/良品判为不良_2022-02-18'
        ]

        reinforce_dir = '/home/cjkai/workspace/dataset/MagnetRing/reinforce'

        # copy ok data
        ok_dir = os.path.join(target_dir, 'OK')
        if os.path.exists(ok_dir):
            shutil.rmtree(ok_dir)
        os.mkdir(ok_dir)
        for origin_dir in origin_dirs:
            copy_ok_data(origin_dir)
        
        # create
        train_dir = os.path.join(target_dir, 'train')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        else:
            shutil.rmtree(train_dir)
            os.mkdir(train_dir)
        val_dir = os.path.join(target_dir, 'val')
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
        else:
            shutil.rmtree(val_dir)
            os.mkdir(val_dir)

        # copy
        for origin_dir in origin_dirs:
            copy_file(origin_dir, ratio=0.9)

        # labelImg to yolo
        image_files = [item for item in os.listdir(train_dir) if item.split('.')[-1] == image_suffix]
        image_files = [os.path.join(train_dir, item) for item in image_files]
        image_files.extend([os.path.join(val_dir, item) for item in os.listdir(val_dir) if item.split('.')[-1] == image_suffix])
        
        json_files = [item.replace(image_suffix, label_suffix) for item in image_files]
        labelImg2yolo(image_files, json_files)

        # empty box
        for origin_dir in origin_dirs:
            create_empty_txt(origin_dir)

        # reinforce data
        create_reinforce()

        train_cache = os.path.join(target_dir, 'train.cache')
        if os.path.isfile(train_cache):
            os.remove(train_cache)
        val_cache = os.path.join(target_dir, 'val.cache')
        if os.path.isfile(val_cache):
            os.remove(val_cache)

    elif args.project == 1:
        label2num = {'NG':0}

        label_suffix = 'json'
        image_suffix = 'jpg'

        target_dir = '/home/cjkai/workspace/dataset/Tissue/yolo'

        origin_dirs = [
            '/home/cjkai/workspace/dataset/Tissue/yolo/2022-01-23',
            '/home/cjkai/workspace/dataset/Tissue/yolo/2022-01-26-2022-01-28',
            '/home/cjkai/workspace/dataset/Tissue/yolo/2022-01-29',
            '/home/cjkai/workspace/dataset/Tissue/yolo/2022-01-31',
            '/home/cjkai/workspace/dataset/Tissue/yolo/2022-02-16'
        ]

        # copy ok data
        ok_dir = os.path.join(target_dir, 'OK')
        if os.path.exists(ok_dir):
            shutil.rmtree(ok_dir)
        os.mkdir(ok_dir)
        for origin_dir in origin_dirs:
            copy_ok_data(origin_dir)
        
        # create
        train_dir = os.path.join(target_dir, 'train')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        else:
            shutil.rmtree(train_dir)
            os.mkdir(train_dir)
        val_dir = os.path.join(target_dir, 'val')
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
        else:
            shutil.rmtree(val_dir)
            os.mkdir(val_dir)

        # copy
        for origin_dir in origin_dirs:
            copy_file(origin_dir, ratio=0.9)

        # labelImg to yolo
        image_files = [item for item in os.listdir(train_dir) if item.split('.')[-1] == image_suffix]
        image_files = [os.path.join(train_dir, item) for item in image_files]
        image_files.extend([os.path.join(val_dir, item) for item in os.listdir(val_dir) if item.split('.')[-1] == image_suffix])
        
        json_files = [item.replace(image_suffix, label_suffix) for item in image_files]
        labelImg2yolo(image_files, json_files)

        # empty box
        for origin_dir in origin_dirs:
            create_empty_txt(origin_dir)

        train_cache = os.path.join(target_dir, 'train.cache')
        if os.path.isfile(train_cache):
            os.remove(train_cache)
        val_cache = os.path.join(target_dir, 'val.cache')
        if os.path.isfile(val_cache):
            os.remove(val_cache)