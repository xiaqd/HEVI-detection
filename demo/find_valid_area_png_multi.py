import os
from PIL import Image
import numpy as np
import cv2
from multiprocessing import Pool
Image.MAX_IMAGE_PIXELS=None

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
def filter_by_area(mask_array, conf_thresh, area_thresh):

    valid_inter_array = mask_array >= conf_thresh

    final_mask = valid_inter_array * 255

    final_mask = final_mask.astype(np.uint8)


    contours, hierarchy = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours_mask = np.zeros(valid_inter_array.shape, dtype=np.uint8)

    valid_contours_list = []
    for contour in contours:
        if cv2.contourArea(contour) > area_thresh:
            
            valid_contours_list.append(contour)


    cv2.drawContours(valid_contours_mask, valid_contours_list, -1, (1), -1)



    final_mask_array = valid_contours_mask * mask_array
    return final_mask_array, len(contours), len(valid_contours_list)

def process_file(tmp_img, input_png_path, input_mask_hev_path, input_mask_tumor_path, output_hev_dir, output_tumor_dir, output_inter_dir, 
                 color_list, tumor_conf_thresh, hev_conf_thresh, inter_conf_thresh, hev_area_thresh, inter_area_thresh):
        
        # print('process file {}'.format(tmp_img))
        img_basename = os.path.splitext(tmp_img)[0]
        tmp_img_path = os.path.join(input_png_path, tmp_img)

        image_array_obj = Image.open(tmp_img_path)


        mask_hev_array = np.array(Image.open(os.path.join(input_mask_hev_path, tmp_img)))
        mask_hev_array = mask_hev_array.astype(np.float32)
        mask_hev_array = mask_hev_array/255
        mask_hev_array, old_hev_count, new_hev_count = filter_by_area(mask_hev_array, conf_thresh=hev_conf_thresh, area_thresh=hev_area_thresh)


        mask_tumor_array = np.array(Image.open(os.path.join(input_mask_tumor_path, tmp_img)))
        mask_tumor_array = mask_tumor_array.astype(np.float32)
        mask_tumor_array = mask_tumor_array/255
        valid_tumor_array = mask_tumor_array > tumor_conf_thresh
        mask_tumor_array = valid_tumor_array * mask_tumor_array


        mask_inter_array = mask_hev_array * mask_tumor_array


        mask_inter_array, old_hevi_count, new_hevi_count = filter_by_area(mask_inter_array, conf_thresh=inter_conf_thresh, area_thresh=inter_area_thresh)

        print('### file {}: old hev count: {}, new hev count: {}, old hevi count: {}, new hevi count: {}'.format(tmp_img, str(old_hev_count), str(new_hev_count),
                                                                                                                          str(old_hevi_count), str(new_hevi_count)))

        blend_inter_obj = generate_color_blend_img(image_array_obj, mask_inter_array, color_list[2])
        blend_hev_obj = generate_color_blend_img(image_array_obj, mask_hev_array, color_list[0])
        blend_tumor_obj = generate_color_blend_img(image_array_obj, mask_tumor_array, color_list[1])


        output_inter_path = os.path.join(output_inter_dir, img_basename+'.jpg')
        output_hev_path = os.path.join(output_hev_dir, img_basename+'.jpg')
        output_tumor_path = os.path.join(output_tumor_dir, img_basename+'.jpg')

        blend_inter_obj.save(output_inter_path)
        blend_hev_obj.save(output_hev_path)
        blend_tumor_obj.save(output_tumor_path)

def func_wrapper(params):
    process_file(*params)

def params_generator(img_list, input_png_path, input_mask_hev_path, input_mask_tumor_path, 
                              final_output_prefix_hev_dir, final_output_prefix_tumor_dir, final_output_prefix_inter_dir, 
                              color_list, tumor_conf_thresh, hev_conf_thresh, inter_conf_thresh, hev_area_thresh, inter_area_thresh):
    for img_path in img_list:
        yield [img_path, input_png_path, input_mask_hev_path, input_mask_tumor_path, 
                        final_output_prefix_hev_dir, final_output_prefix_tumor_dir, final_output_prefix_inter_dir, 
                        color_list, tumor_conf_thresh, hev_conf_thresh, inter_conf_thresh, hev_area_thresh, inter_area_thresh]


if __name__ == '__main__':

# params setting

    # the image related mask path
    input_mask_hev_path = 'hev'
    input_mask_tumor_path = 'tumor'

    # the png image path
    input_png_path = 'images'

    hev_area_thresh = 150
    inter_area_thresh = 20

    tumor_conf_thresh = 0.9
    hev_conf_thresh = 0.9
    inter_conf_thresh = 0.9

    output_prefix_hev_dir = 'demo_results/hev'
    output_prefix_tumor_dir = 'demo_results/tumor'
    output_prefix_inter_dir = 'demo_results/inter'

    color_list = [[255,128,128],[128,255,128],[0, 255, 220]]

    num_proc = 8
    
    # params setting done
    print('### thresh setting: ###')
    print('tumor_conf_thresh: {}'.format(str(tumor_conf_thresh)))
    print('hev_conf_thresh: {}'.format(str(hev_conf_thresh)))
    print('inter_conf_thresh: {}'.format(str(inter_conf_thresh)))
    print('hev_area_thresh: {}'.format(str(hev_area_thresh)))
    print('inter_area_thresh: {}'.format(str(inter_area_thresh)))
    print('### thresh setting done ###')
    p = Pool(num_proc)

    final_output_prefix_hev_dir = output_prefix_hev_dir + '_hconf_{}_tconf_{}_iconf_{}_harea_{}_iarea_{}'.format(str(hev_conf_thresh), str(tumor_conf_thresh),
                                                                                                str(inter_conf_thresh),
                                                                                                str(hev_area_thresh),str(inter_area_thresh))
    final_output_prefix_tumor_dir = output_prefix_tumor_dir + '_hconf_{}_tconf_{}_iconf_{}_harea_{}_iarea_{}'.format(str(hev_conf_thresh), str(tumor_conf_thresh),
                                                                                                str(inter_conf_thresh),
                                                                                                str(hev_area_thresh),str(inter_area_thresh))
    final_output_prefix_inter_dir = output_prefix_inter_dir + '_hconf_{}_tconf_{}_iconf_{}_harea_{}_iarea_{}'.format(str(hev_conf_thresh), str(tumor_conf_thresh),
                                                                                                str(inter_conf_thresh),
                                                                                                str(hev_area_thresh),str(inter_area_thresh))
    
    if not os.path.isdir(final_output_prefix_hev_dir):
        os.makedirs(final_output_prefix_hev_dir)
    if not os.path.isdir(final_output_prefix_tumor_dir):
        os.makedirs(final_output_prefix_tumor_dir)
    if not os.path.isdir(final_output_prefix_inter_dir):
        os.makedirs(final_output_prefix_inter_dir)

    img_list = scan_files(input_png_path, ext_list=['.png',], replace_root=True)

    p.map(func_wrapper, params_generator(img_list, input_png_path, input_mask_hev_path, input_mask_tumor_path, 
                                                   final_output_prefix_hev_dir, final_output_prefix_tumor_dir, final_output_prefix_inter_dir, 
                                                   color_list, tumor_conf_thresh, hev_conf_thresh, inter_conf_thresh, hev_area_thresh, inter_area_thresh))





