import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


DATASET_PATH = 'D:/data/Data/'
SAVE_PATH = 'D:/data/Resize_frames/img_' 

def center_crop(img, set_size):

    h, w, c = img.shape

    if set_size > min(h, w):
        return img

    crop_width = set_size
    crop_height = set_size

    mid_x, mid_y = w//2, h//2
    offset_x, offset_y = crop_width//2, crop_height//2
       
    crop_img = img[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]

    return crop_img


if __name__ == "__main__":

    for i in os.listdir(DATASET_PATH):
        
        print("Current Index : " + i)
        
        frame_number = '000'
        
        if i == '169':
            frame_number = '020'
        elif i== '62':
            frame_number = '016'
        elif i== '90':
            frame_number = '013'
        elif i== '102':
            frame_number = '019'
        elif i== '115':
            frame_number = '026'
        elif i== '118':
            frame_number = '009'
        elif i== '121':
            frame_number = '023'
        elif i== '139':
            frame_number = '028'
        elif i== '171':
            frame_number = '015'
        
        img_path = DATASET_PATH + str(i) + '/rgb_images/' + frame_number + '.jpg'
        
        
        img = cv2.imread(img_path)
        h, w, c = img.shape
        
        crop_size = 1000            
        img = center_crop(img, crop_size)
        img = cv2.resize(img, (300, 450))

        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        
        cv2.imwrite(SAVE_PATH + "%03d.jpg" %int(i) , img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()