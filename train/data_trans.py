import argparse
from utils.general import (Logging, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer)
import os
from all2yolo import all2yolo,convert
import random
import xml.etree.ElementTree as ET




def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    

    # Weights & Biases argume
    parser.add_argument('--save_dir', default='/workspace/volume/guojun/Train/ObjDetection/outputs', help='runs/train save to project/name')   # ROOT / 'runs'
    # SDK
    parser.add_argument('--dataset_path', type=str, required=True, help='train dataset path')
    parser.add_argument("--pretrained_model", type=str, default=None, help="pretrained model path")
    parser.add_argument('--train_ratio', type=float, default=0.9, help='train dataset proportion')
    parser.add_argument("--label_name", nargs="+", required=True, help='category order')
    parser.add_argument("--save_model", type=str, default=None, help="save model path")

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
opt = parse_opt()
classes = opt.label_name
print("xml")
files_list = os.listdir(opt.dataset_path) 
filter_files_list = [fn for fn in files_list if fn.endswith("xml")]
num = len(filter_files_list)

ftrain1 = open(opt.dataset_path+'/name.txt', 'w')
for i in range(num):
    name = filter_files_list[i][:-4] + '\n'
    ftrain1.write(name)
ftrain1.close()

print('SPlit data train and val ')
num = len(filter_files_list)  #统计所有的标注文件
list_num = range(num)
tv = int(num * 0.1)  # 设置训练验证集的数目
tr = int(tv * 0.9)      # 设置训练集的数目
trainval = random.sample(list_num, tv)
train = random.sample(trainval, tr)

# txt 文件写入的只是xml 文件的文件名（数字），没有后缀，如下图。
ftrainval = open(opt.save_dir+'/trainval.txt', 'w')
ftest = open(opt.save_dir+'/test.txt', 'w')
ftrain = open(opt.save_dir+'/train.txt', 'w')
fval = open(opt.save_dir+'/val.txt', 'w')

for i in list_num:
    name = opt.dataset_path+"/"+filter_files_list[i][:-4]+'.jpg' + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
print('xml to coco txt ')

image_ids = open(opt.dataset_path+'/name.txt').read().strip().split()
for image_id in image_ids:
    in_file = open(opt.dataset_path+'/%s.xml' % (image_id))
    out_file = open(opt.dataset_path+'/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
            float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
print("yes")