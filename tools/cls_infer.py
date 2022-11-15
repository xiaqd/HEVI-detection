import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from albumentations.core.composition import Compose
from albumentations.augmentations import transforms, geometric
from albumentations.pytorch.transforms import ToTensorV2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def scan_files(input_file_path, ext_list = ['.txt'], replace_root=True):
    file_list = []
    for root, dirs, files in os.walk(input_file_path):
        # scan all files and put it into list

        for f in files:
            if os.path.splitext(f)[1].lower() in ext_list:
                if replace_root:
                    file_list.append(os.path.join(root, f).replace("\\","/").replace(os.path.join(input_file_path, "").replace("\\","/"), "", 1 ))
                else:
                    file_list.append(os.path.join(root, f).replace("\\","/"))

    return file_list

def get_model(input_model_path, num_classes):
    model = timm.create_model("resnest14d", pretrained=False, num_classes=num_classes)

    model = nn.DataParallel(model)
    model.to('cuda:0')

    model.load_state_dict(torch.load(input_model_path))
    
    model.eval()


    return model



def main(input_img_path, input_model_path, img_rgb_mean, img_rgb_std, input_size, class_name_list):
    cls_num = len(class_name_list)

    model = get_model(input_model_path, cls_num)

    preprocess = Compose([
        geometric.resize.Resize(input_size, input_size),
        transforms.Normalize(mean=img_rgb_mean, std=img_rgb_std),
        ToTensorV2(),
    ])


    img = cv2.imread(input_img_path)[:,:,::-1]

    tensor_img = preprocess(image=img)['image'].unsqueeze(0).cuda()

    output = model(tensor_img)
    output = torch.softmax(output.data, dim=1)
    result = output.cpu().detach().numpy()[0]
    for cls_name, cls_prob in zip(class_name_list, result):
        print('{} result:'.format(input_img_path))
        print('{}:{}'.format(cls_name, str(cls_prob)))


if __name__ == '__main__':
    input_img_path = 'path/to/image/path'
    input_model_path = 'path/to/model'


    img_rgb_mean = [0.813, 0.634, 0.733]
    img_rgb_std = [0.091, 0.145, 0.108]
    input_size = 512
    class_name_list = ['neg', 'pos']


    main(input_img_path, input_model_path, img_rgb_mean, img_rgb_std, input_size, class_name_list)
