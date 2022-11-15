import os

from torch.utils.data import Dataset
import torch
import cv2


class ClsDataset(Dataset):
    def __init__(self, root_dir, img_path_list, label_list, transform=None):
        self.root_dir = root_dir
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        
        img = cv2.imread(os.path.join(self.root_dir, self.img_path_list[index]))[:,:,:3]
        img = img[:,:,::-1]
        y_label = torch.tensor(int(self.label_list[index]),dtype=torch.long)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return (img, y_label)