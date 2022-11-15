import os
import random

import cv2
import numpy as np
import torch



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



class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, class_name_list, img_ext='.png', mask_ext='.png', target_patch_num=5000, target_image_size=[512, 1024], area_thresh=0.1, pure_bg_ratio=0.1, mode='multilabel', transform=None, preprocessing=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.class_name_list = class_name_list
        self.num_classes = len(self.class_name_list)
        self.transform = transform
        self.preprocessing = preprocessing
        self.mode = mode
        self.target_path_num = target_patch_num
        self.target_img_size = target_image_size
        self.pure_bg_ratio = pure_bg_ratio


        self.img_ids = scan_files(self.img_dir, ext_list = [img_ext])
        self.img_ids = np.array(self.img_ids)

        self.area_thresh = area_thresh
            
    def __len__(self):
        return self.target_path_num

    def __getitem__(self, idx):

        # randomly select a region image to crop
        true_idx = random.randint(0,len(self.img_ids)-1)
        whole_img_array = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, self.img_ids[true_idx])), cv2.COLOR_BGR2RGB)
        img_id = self.img_ids[true_idx]

        h,w = whole_img_array.shape[:2]

        current_target_size = min(h,w)


        whole_mask_array = np.zeros((h,w,self.num_classes))
        for i, label_name in enumerate(self.class_name_list):
            mask_path = os.path.join(self.mask_dir, label_name, os.path.splitext(img_id)[0]+self.mask_ext)
            whole_mask_array[:,:,i] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        

        # randomly crop a patch in selected region image
        while True:
            current_target_img_size = random.randint(min(self.target_img_size[0], current_target_size), min(self.target_img_size[1], current_target_size))
            
            y_range = h - current_target_img_size
            x_range = w - current_target_img_size

            start_x = random.randint(0, max(0,x_range-1))
            start_y = random.randint(0, max(0,y_range-1))
            end_x = start_x + current_target_img_size
            end_y = start_y + current_target_img_size
            patch_mask_array = np.zeros((current_target_img_size, current_target_img_size, self.num_classes))
            for i in range(self.num_classes):
                patch_mask_array[:,:,i] = np.array(whole_mask_array[start_y:end_y, start_x:end_x, i])
            

            current_area_ratio = np.sum(patch_mask_array>0)/(patch_mask_array.shape[0]*patch_mask_array.shape[1])
            if current_area_ratio > self.area_thresh:
                img = np.array(whole_img_array[start_y:end_y, start_x:end_x])
                break
        

        mask = patch_mask_array

        ori_img = np.array(img)

        # data augmentation
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
        else:
            img = img.astype('float32') / 255
            img = img.transpose(2, 0, 1)

        if self.mode != 'multiclass':
            mask = mask.astype('float32') / 255
        

        ori_img = cv2.resize(ori_img, (img.shape[2], img.shape[1]) )
        ori_img = ori_img.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}, ori_img
