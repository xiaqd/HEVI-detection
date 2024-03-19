# HEVI Detection
### Note: This repository contains part of the code for this paper. 
----------
## Segmentation Training 
Prepare the related images and masks first.

To train a model, using code in **seg_training folder**.

The example below is showing how to train a model using the code in seg_training folder:
```
python3 train.py --num_classes 2 \
                 --dataset_root dataset_root_path \
                 --dataset dataset_dir_name \
                 --lr 0.001 \
                 --epochs 1000 \
                 --name hev_exp_1 \
                 --optimizer RAdam \
                 --input_w 512 \
                 --input_h 512 \
                 -b 16 \
                 --num_workers 36 \
                 --class_name hev,tumor \
                 --show_arch False \
                 --mix_precision True \
                 --encoder timm-resnest14d \
                 --encoder_weight histology \
                 --arch unetplusplus \
                 --mode multilabel \
                 --tf_log_path tblog_dir/hevi_seg/hev_exp_1_log \
                 --loss tversky \
                 --loss_weight 1.0 \
                 --scheduler CosineAnnealingWarmupRestarts \
                 --act sigmoid \
                 --cosine_cycle_steps 50 \
                 --cosine_cycle_warmup 5 \
                 --cosine_cycle_gamma 0.95 \
                 --early_stopping 50
```
#### Note: you might need to change encoder_weight from **histology** to **imagenet**.

----------
## Requirements
- Python 3.6
- Pytorch 1.7
- Torchvision 0.8
- segmentation_models_pytorch 0.2.1
- aicspylibczi 3.0.1
----------

## Segmentation Inference

To infer images, you could refer to **infer_regions.py** in **tools** folder. You may need to modify these part of code in order to run your own data and model. You may need to put segmentation_models_pytorch in the current directory or pip install it to run the code.
``` python
if __name__ == '__main__':
    input_img_path = 'input_img_dir'
    input_model_path = 'path/to/model'
    input_model_config_path = 'path/to/model_config'

    output_mask_path = 'output_result_dir'

    img_rgb_mean = [0.813, 0.634, 0.733] # rgb mean based on our histology images
    img_rgb_std = [0.091, 0.145, 0.108] # rgb std based on our histology images
    crop_size = [512,768,1024]
    overlap = [384,576,768]
```
#### Note: To **handle CZI format**, you could refer to **czi_reader.py** and **filter_valid_area.py** in **tools** folder, the details of the code about infering a whole CZI file is not provided in this repository.
----------
## Classification Training
To train classification model, using code in **cls_training folder**.
The example below is showing how to train a model using the code in cls_training folder:
```
python3 train.py dataset_path 
```

the structure in the dataset_path should be like this:
```
-dataset_path
--wsi_name_1
---0
----img1
----...
----imgN
---1
----img1
----...
----imgN
.
.
.
--wsi_name_N
---0
----img1
----...
----imgN
---1
----img1
----...
----imgN
```
----------
## Classification Inference
To infer images, you could refer to **cls_infer.py** in **tools** folder. You may need to modify these part of code in order to run your own data and model.

``` python
if __name__ == '__main__':
    input_img_path = 'path/to/image/path'
    input_model_path = 'path/to/model'


    img_rgb_mean = [0.813, 0.634, 0.733]
    img_rgb_std = [0.091, 0.145, 0.108]
    input_size = 512
    class_name_list = ['neg', 'pos']
```
----------
## Finding and Visualizing HEVI area
**filter_valid_area.py** in **tools** folder is the example how to find and visualize HEVI in a scene in CZI image. 
``` python
input_mask_path = '/path/to/pred_dir' # input pred mask folder, should be a png file
input_czi_path = '/path/to/czi_file' # input czi image path
scene = 1 # czi scene number
output_prefix = 'output_dir' # output folder


hev_area_thresh = 150 # valid hev pixel area
inter_area_thresh = 20 # valid hevi pixel area

tumor_conf_thresh = 0.99 # minimum confidence of valid tumor pixels 
hev_conf_thresh = 0.99 # minimum confidence of valid HEV pixels 
inter_conf_thresh = 0.99 # minimum confidence of valid HEVI pixels 


color_list = [[255,128,128],[128,255,128],[0, 255, 220]] # mask color for blend images, only using the last one for HEVI
```
----------
## DEMO
This demo show how to use segmentation predication result to find HEVI area in a patch of a whole slide image. 
Run the code below to generate a blend image with HEVI area marked.

```
python3 find_valid_area_png_multi.py
```

You could find the blend image results in the the folder **demo_results**.

More details settings could be found in the code **find_valid_area_png_multi.py**.

----------
## Reference
We modified a little code from this library: https://github.com/qubvel/segmentation_models.pytorch

Our training code refer to this repository: https://github.com/milesial/Pytorch-UNet

We using RAdam in this repository: https://github.com/LiyuanLucasLiu/RAdam
