import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import pickle
import yaml
from albumentations.core.composition import Compose
from albumentations.augmentations import transforms, geometric
from albumentations.pytorch.transforms import ToTensorV2
import segmentation_models_pytorch as smp

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
ARCH_DICT = {'unetplusplus': smp.UnetPlusPlus,
             'unet': smp.Unet,
             'fpn': smp.FPN,
             'pspnet': smp.PSPNet,
             'pan': smp.PAN,
             'manet': smp.MAnet,
             'linknet': smp.Linknet,
             'deeplabv3': smp.DeepLabV3,
             'deeplabv3plus': smp.DeepLabV3Plus
            }

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        content = pickle.load(f)
    return content


def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        content = yaml.load(f)
    return content    


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

def get_model(input_model_path, input_model_config_path):
    config = load_yaml(input_model_config_path)

    input_width = config['input_w']
    input_height = config['input_h']
    class_name_list = config['class_name']

    model = ARCH_DICT[config['arch']](
        encoder_name=config['encoder'],
        encoder_weights=config['encoder_weight'],
        classes=config['num_classes'],
        activation=config['act'],
    )

    model = nn.DataParallel(model)
    model.to('cuda:0')

    model.load_state_dict(torch.load(input_model_path))
    
    model.eval()


    return model, input_width, input_height, class_name_list


def get_crop_area(width, height, crop_size, overlap):
    current_crop_width = min(width, crop_size)
    current_crop_height = min(height, crop_size)
    assert crop_size > overlap, 'crop size must great than overlap: {} : {}'.format(str(crop_size),str(overlap))
    regions = []

    h_start = 0
    while h_start < height:
        w_start = 0
        while w_start < width:
            region_x2 = min(max(0, w_start + current_crop_width), width)
            region_y2 = min(max(0, h_start + current_crop_height), height)
            region_x1 = min(max(0, region_x2 - current_crop_width), width)
            region_y1 = min(max(0, region_y2 - current_crop_height), height)

            regions.append([region_x1, region_y1, region_x2, region_y2])

            # break when region reach the end
            if w_start + current_crop_width >= width: break

            w_start += current_crop_width - overlap

        # break when region reach the end
        if h_start + current_crop_height >= height: break

        h_start += current_crop_height - overlap

    regions = np.array(regions, dtype=int)
    return regions


def main(input_img_path, input_model_path, input_model_config_path, output_mask_path, img_rgb_mean, img_rgb_std, crop_size_list, overlap_list, class_thres=[0.5, 0.5]):

    model, input_width, input_height, class_name_list = get_model(input_model_path, input_model_config_path)

    cls_num = len(class_name_list)

    preprocess = Compose([
        geometric.resize.Resize(input_height, input_width),
        transforms.Normalize(mean=img_rgb_mean, std=img_rgb_std),
        ToTensorV2(),
    ])


    img_file_list = scan_files(input_img_path, ext_list = ['.png'])

    for f_path in img_file_list:
        print('process image: {}'.format(f_path))
        img_path = os.path.join(input_img_path, f_path)
        whole_img = cv2.imread(img_path)[:,:,::-1]

        count_mask = np.zeros(whole_img.shape[:2], dtype=int)
        whole_mask = np.zeros(tuple([cls_num]+list(whole_img.shape[:2])), dtype=float)

        for tmp_crop_size, tmp_overlap in zip(crop_size_list, overlap_list):

            regions = get_crop_area(whole_img.shape[1], whole_img.shape[0], tmp_crop_size, tmp_overlap)

            for region in regions:
                x1, y1, x2, y2 = region

                tiled_img = np.array(whole_img[y1:y2,x1:x2])

                tensor_img = preprocess(image=tiled_img)['image'].unsqueeze(0).cuda()

                current_tiled_mask= model(tensor_img)
                current_tiled_mask = F.upsample(current_tiled_mask, (y2-y1, x2-x1), mode='bilinear')

                tiled_mask = current_tiled_mask.cpu().detach().numpy()[0]

                count_mask[y1:y2,x1:x2] = count_mask[y1:y2,x1:x2] + 1
                whole_mask[:,y1:y2,x1:x2] = whole_mask[:,y1:y2,x1:x2] + tiled_mask
            
        final_whole_mask = whole_mask/count_mask

        
        inter_output_path = os.path.join(output_mask_path, 'inter', os.path.splitext(f_path)[0]+'.png')
        inter_mask = np.ones(whole_img.shape[:2], dtype=int) 

        for i, sub_whole_mask in enumerate(final_whole_mask[:]):

            current_mask_output_path = os.path.join(output_mask_path, 'mask', class_name_list[i], os.path.splitext(f_path)[0]+'.png')

            if not os.path.isdir(os.path.dirname(current_mask_output_path)):
                os.makedirs(os.path.dirname(current_mask_output_path))
            inter_mask = (sub_whole_mask>class_thres[i]) & inter_mask
            output_mask = np.where(sub_whole_mask>class_thres[i], 255, 0)
            output_mask = output_mask.astype(np.uint8)
            cv2.imwrite(current_mask_output_path, output_mask)

        if not os.path.isdir(os.path.dirname(inter_output_path)):
            os.makedirs(os.path.dirname(inter_output_path))
        inter_mask = inter_mask*255
        inter_mask = inter_mask.astype(np.uint8)
        cv2.imwrite(inter_output_path, inter_mask)        
        
 

if __name__ == '__main__':
    input_img_path = 'input_img_dir'
    input_model_path = 'path/to/model'
    input_model_config_path = 'path/to/model_config'

    output_mask_path = 'output_result_dir'

    img_rgb_mean = [0.813, 0.634, 0.733]
    img_rgb_std = [0.091, 0.145, 0.108]
    crop_size = [512,768,1024]
    overlap = [384,576,768]

    main(input_img_path, input_model_path, input_model_config_path, output_mask_path, img_rgb_mean, img_rgb_std, crop_size, overlap)
