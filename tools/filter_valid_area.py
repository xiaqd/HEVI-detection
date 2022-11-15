import os
from PIL import Image
import numpy as np
import cv2

# remove image limitation in PIL
Image.MAX_IMAGE_PIXELS=None

from czi_reader import CziObject

def create_color_mask(width, height, color_value):
    mask = np.ones((height, width, len(color_value)))
    for i, sc in enumerate(color_value):
        mask[:,:,i] = mask[:,:,i]*sc
    return mask.astype(np.uint8)


def generate_color_blend_img(img_pil_obj, pred_mask, color_value, alpha=0.5):
    width, height = img_pil_obj.size
    color_mask = create_color_mask(width, height, color_value)

    hsv_color_array = cv2.cvtColor(color_mask, cv2.COLOR_RGB2HSV)
    hsv_color_array[:,:,2] = hsv_color_array[:,:,2]*pred_mask
    rbg_color_array = cv2.cvtColor(hsv_color_array, cv2.COLOR_HSV2RGB)
    
    final_mask_pil_obj = Image.fromarray(rbg_color_array).convert('RGB')

    out_blend_obj = Image.blend(img_pil_obj, final_mask_pil_obj, alpha=alpha)
    return out_blend_obj


# mask_array: value from 0 to 1, float
def filter_by_area(mask_array, conf_thresh, area_thresh, preprocss=False):

    valid_inter_array = mask_array >= conf_thresh

    final_mask = valid_inter_array * 255

    final_mask = final_mask.astype(np.uint8)

    if preprocss:
        kernel1 = np.ones((3,3), np.uint8)
        kernel2 = np.ones((3,3), np.uint8)

        final_mask = cv2.dilate(final_mask, kernel1, iterations=12)
        final_mask = cv2.erode(final_mask, kernel2, iterations=10)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours_mask = np.zeros(valid_inter_array.shape, dtype=np.uint8)

    print('ori contours num:{}'.format(str(len(contours))))

    valid_contours_list = []
    for contour in contours:
        if cv2.contourArea(contour) > area_thresh:
            
            valid_contours_list.append(contour)

    print('new contours num:{}'.format(str(len(valid_contours_list))))

    cv2.drawContours(valid_contours_mask, valid_contours_list, -1, (1), -1)

    final_mask_array = valid_contours_mask * mask_array
    return final_mask_array

input_mask_path = '/path/to/pred_dir' # input pred mask folder, should be a png file
input_czi_path = '/path/to/czi_file' # input czi image path
scene = 1 # czi scene number
output_prefix = 'output_dir' # output folder


input_mask_hev_path = os.path.join(input_mask_path, 'hev', '{}_scene_{}.png'.format(os.path.splitext(os.path.basename(input_czi_path))[0], str(scene)))
input_mask_tumor_path = os.path.join(input_mask_path, 'tumor', '{}_scene_{}.png'.format(os.path.splitext(os.path.basename(input_czi_path))[0], str(scene)))


hev_area_thresh = 150 # valid hev pixel area
inter_area_thresh = 20 # valid hevi pixel area

tumor_conf_thresh = 0.99 # minimum confidence of valid tumor pixels 
hev_conf_thresh = 0.99 # minimum confidence of valid HEV pixels 
inter_conf_thresh = 0.99 # minimum confidence of valid HEVI pixels 


color_list = [[255,128,128],[128,255,128],[0, 255, 220]] # mask color for blend images, only using the last one for HEVI

os_img_obj = CziObject(input_czi_path)


image_array = os_img_obj.get_thumbnail(scene=scene)
image_array_obj = Image.fromarray(image_array)

mask_hev_array = np.array(Image.open(input_mask_hev_path))
mask_hev_array = mask_hev_array.astype(np.float32)
mask_hev_array = mask_hev_array/255

mask_hev_array = filter_by_area(mask_hev_array, conf_thresh=hev_conf_thresh, area_thresh=hev_area_thresh)


mask_tumor_array = np.array(Image.open(input_mask_tumor_path))
mask_tumor_array = mask_tumor_array.astype(np.float32)
mask_tumor_array = mask_tumor_array/255
valid_tumor_array = mask_tumor_array > tumor_conf_thresh
mask_tumor_array = valid_tumor_array * mask_tumor_array


mask_inter_array = mask_hev_array * mask_tumor_array


print('tumor_conf_thresh: {}'.format(str(tumor_conf_thresh)))
print('hev_conf_thresh: {}'.format(str(hev_conf_thresh)))
print('inter_conf_thresh: {}'.format(str(inter_conf_thresh)))
print('hev_area_thresh: {}'.format(str(hev_area_thresh)))
print('inter_area_thresh: {}'.format(str(inter_area_thresh)))


mask_inter_array = filter_by_area(mask_inter_array, conf_thresh=inter_conf_thresh, area_thresh=inter_area_thresh, preprocss=True)


blend_inter_obj = generate_color_blend_img(image_array_obj, mask_inter_array, color_list[2])



output_img_path = output_prefix + '_hconf_{}_tconf_{}_iconf_{}_harea_{}_iarea_{}.jpg'.format(str(hev_conf_thresh), str(tumor_conf_thresh), 
                                                                                               str(inter_conf_thresh),
                                                                                               str(hev_area_thresh),str(inter_area_thresh))
if not os.path.isdir(os.path.dirname(output_img_path)):
    os.makedirs(os.path.dirname(output_img_path))
blend_inter_obj.save(output_img_path)





