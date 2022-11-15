import aicspylibczi
from aicspylibczi import CziFile
from pathlib import Path

import numpy as np

class CziObject:
    
    def __init__(self, img_path):
        self.reader = CziFile(Path(img_path))        
        self.scene_num = len(self.reader.get_dims_shape())


    def get_scene_num(self):
        return self.scene_num
    
    def get_img_size(self, scene=None):
        if scene is not None:
            return [self.reader.get_scene_bounding_box(scene).w, self.reader.get_scene_bounding_box(scene).h]

        size_list = []
        for i in range(self.scene_num):
            size_list.append([self.reader.get_scene_bounding_box(i).w, self.reader.get_scene_bounding_box(i).h])

        return size_list

    def read_region(self, x, y, w, h, scene, scale_ratio=1.):
        bias_x = self.reader.get_scene_bounding_box(scene).x
        bias_y = self.reader.get_scene_bounding_box(scene).y
        
        img = self.reader.read_mosaic((bias_x+x, bias_y+y, w, h), scale_factor=scale_ratio, background_color=(1.,1.,1.), C=0)

        img = np.squeeze(img)
        img = img[:, :, ::-1]

        # output is numpy array with RGB channel
        return np.array(img)

    def get_thumbnail(self, scene, scale=0.125):
        w, h = self.get_img_size(scene)

        img = np.array(self.read_region(0, 0, w, h, scene, scale_ratio = scale))

        return img

