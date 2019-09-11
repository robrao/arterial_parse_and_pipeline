import os
import pydicom
import csv
import parsing
import glob
import display_imgs

erroneous_masks = [
    './final_data/dicoms/SCD0000201/1.dcm',
    './final_data/dicoms/SCD0000201/12.dcm',
    './final_data/dicoms/SCD0000201/8.dcm',
    './final_data/dicoms/SCD0000201/14.dcm',
    './final_data/dicoms/SCD0000201/6.dcm',
    './final_data/dicoms/SCD0000201/18.dcm',
    './final_data/dicoms/SCD0000201/22.dcm',
    './final_data/dicoms/SCD0000201/16.dcm',
    './final_data/dicoms/SCD0000201/2.dcm',
    './final_data/dicoms/SCD0000301/1.dcm',
    './final_data/dicoms/SCD0000301/2.dcm',
    './final_data/dicoms/SCD0000301/12.dcm',
    './final_data/dicoms/SCD0000301/14.dcm',
    './final_data/dicoms/SCD0000301/16.dcm',
    './final_data/dicoms/SCD0000301/18.dcm',
    './final_data/dicoms/SCD0000401/8.dcm',
    './final_data/dicoms/SCD0000401/18.dcm',
    './final_data/dicoms/SCD0000401/6.dcm',
    './final_data/dicoms/SCD0000401/16.dcm',
    './final_data/dicoms/SCD0000401/2.dcm',
]

def img_name_from_path(path):
    return os.path.basename(path).split('-')[2].strip('0') + '.dcm'

def parse_files(files):
    contour_img_map = {}
    for f in files:
        img_name = img_name_from_path(f)
        contour_img_map[img_name] = parsing.parse_contour_file(f)
    return contour_img_map

def csv_to_dict(filename):
    with open(filename, mode='r') as infile:
        reader = csv.reader(infile)
        dir_map = {rows[0]:rows[1] for rows in reader}
        dir_map.pop('patient_id') # remove unnecessary column headers
        return dir_map

def map_imgs_to_masks():
    img_contour_list = {}
    imgs_masks = {}
    dir_dicom_map = csv_to_dict('final_data/link.csv')

    for key in dir_dicom_map:
        dicom_path = "./final_data/dicoms/{}/".format(key)
        contour_path = "./final_data/contourfiles/{}/i-contours/*.txt".format(dir_dicom_map[key])
        contour_files = glob.glob(contour_path)
        img_contours = parse_files(contour_files)

        for img in img_contours:
            img_path = dicom_path + img
            img_contour_list[img_path] = img_contours[img]

    for img_path in img_contour_list:
        assert os.path.exists(img_path)
        # Should normalize brightness, height, and width
        dcm_data = parsing.parse_dicom_file(img_path)

        if (not dcm_data): continue

        mask = parsing.poly_to_mask(img_contour_list[img_path], dcm_data['width'], dcm_data['height'])
        imgs_masks[img_path] = {}
        imgs_masks[img_path]['mask'] = mask
        imgs_masks[img_path]['image'] = dcm_data['pixel_data']
        imgs_masks[img_path]['img_width'] = dcm_data['width']
        imgs_masks[img_path]['img_height'] = dcm_data['height']
        imgs_masks[img_path]['mask_erroneous'] = any(img_path in err_img for err_img in erroneous_masks)

    return imgs_masks


if __name__ == "__main__":
    imgs_masks = map_imgs_to_masks()
    for img_path in imgs_masks:
        mask = imgs_masks[img_path]['mask']
        image = imgs_masks[img_path]['image']

        if (not any(img_path in err_img for err_img in erroneous_masks)):
            display_imgs.display_img_with_mask(image, mask, img_path)
