
import time
import random
import os
import cv2
import numpy as np
import pandas as pd
import gc

from PIL import Image
from openslide import OpenSlide, OpenSlideUnsupportedFormatError

import xml.etree.cElementTree as ET
from shapely.geometry import box, Point, Polygon

from matplotlib import pyplot as plt
import matplotlib.patches as patches

from ast import literal_eval

import pickle


PATCH_SIZE = 500
CHANNEL = 3
CLASS_NUM = 2

DROPOUT = 0.5

THRESH = 90

PIXEL_WHITE = 255
PIXEL_TH = 200
PIXEL_BLACK = 0


SPLIT = 4

level = 1
mag_factor = pow(2, level)



def openSlide_init(tif_file_path, level):
    
    try:
        wsi_obj = OpenSlide(tif_file_path)

    except OpenSlideUnsupportedFormatError:
        print('Exception')
        return None
    else:
        slide_w_, slide_h_ = wsi_obj.level_dimensions[level]
        print('level' + str(level), 'size(w, h):', slide_w_, slide_h_)
        
        return wsi_obj


def read_wsi(wsi_obj, level, mag_factor, sect):
    
    
    
    time_s = time.time()
            
   

   
    # level1 dimension
    width_whole, height_whole = wsi_obj.level_dimensions[level]
    print("level1 dimension (width, height): ", width_whole, height_whole)

    # section size after split
    width_split, height_split = width_whole // SPLIT, height_whole // SPLIT
    print("section size (width, height): ", width_split, height_split)

    delta_x = int(sect[0]) * width_split
    delta_y = int(sect[1]) * height_split

    rgba_image_pil = wsi_obj.read_region((delta_x * mag_factor, \
                                          delta_y * mag_factor), \
                                          level, (width_split, height_split))

    print("rgba image dimension (width, height):", rgba_image_pil.size)

    
    rgba_image = np.asarray(rgba_image_pil)
    print("transformed:", rgba_image.shape)

    time_e = time.time()
    
    print("Time spent on loading WSI section into memory: ", (time_e - time_s))
    
    return rgba_image

def construct_colored_wsi(rgba_):

   
    r_, g_, b_, a_ = cv2.split(rgba_)
    
    wsi_rgb_ = cv2.merge((r_, g_, b_))
    wsi_gray_ = cv2.cvtColor(wsi_rgb_,cv2.COLOR_RGB2GRAY)
    wsi_hsv_ = cv2.cvtColor(wsi_rgb_, cv2.COLOR_RGB2HSV)
    
    return wsi_rgb_, wsi_gray_, wsi_hsv_


def get_contours(cont_img, rgb_image_shape):
    
    
    
    print('contour image dimension: ',cont_img.shape)
    
    contour_coords = []
    _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boundingBoxes = [cv2.boundingRect(c) for c in contours]

    for contour in contours:
        contour_coords.append(np.squeeze(contour))
        
    mask = np.zeros(rgb_image_shape, np.uint8)
    
    print('mask image dimension: ', mask.shape)
    cv2.drawContours(mask, contours, -1, \
                    (PIXEL_WHITE, PIXEL_WHITE, PIXEL_WHITE),thickness=-1)
    
    return boundingBoxes, contour_coords, contours, mask


def segmentation_hsv(wsi_hsv_, wsi_rgb_):
    
    print("HSV segmentation step")
    contour_coord = []
    
    lower_ = np.array([20,20,20])
    upper_ = np.array([200,200,200]) 

    # HSV image threshold
    thresh = cv2.inRange(wsi_hsv_, lower_, upper_)
    
    # print("Closing step: ")
    close_kernel = np.ones((15, 15), dtype=np.uint8) 
    image_close = cv2.morphologyEx(np.array(thresh),cv2.MORPH_CLOSE, close_kernel)
    # print("image_close size", image_close.shape)

   
    # print("Openning step: ")
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, open_kernel)
    # print("image_open size", image_open.size)

    # print("Getting Contour: ")
    bounding_boxes, contour_coords, contours, mask \
    = get_contours(np.array(image_open), wsi_rgb_.shape)
      
    return bounding_boxes, contour_coords, contours, mask


def construct_bags(wsi_obj, wsi_rgb, contours, mask, level, \
                   mag_factor, sect, patch_size, split_num):
    
   

    patches = list()
    patches_coords = list()
    patches_coords_local = list()

    start = time.time()
    
    # level1 dimension
    width_whole, height_whole = wsi_obj.level_dimensions[level]
    width_split, height_split = width_whole // split_num, height_whole // split_num
    # print(width_whole, height_whole)

    # section size after split
    # print(int(sect[0]), int(sect[1]))
    delta_x = int(sect[0]) * width_split
    delta_y = int(sect[1]) * height_split
    # print("delta:", delta_x, delta_y)

    contours_ = sorted(contours, key = cv2.contourArea, reverse = True)
    contours_ = contours_[:5]

    for i, box_ in enumerate(contours_):

        box_ = cv2.boundingRect(np.squeeze(box_))
        # print('region', i)
        # 
        
        b_x_start = int(box_[0])
        b_y_start = int(box_[1])
        b_x_end = int(box_[0]) + int(box_[2])
        b_y_end = int(box_[1]) + int(box_[3])
       
        # step size: PATCH_SIZE / 2 -> PATCH_SIZE

        X = np.arange(b_x_start, b_x_end, step=patch_size)
        Y = np.arange(b_y_start, b_y_end, step=patch_size)        
        
        # print('ROI length:', len(X), len(Y))
        
        for h_pos, y_height_ in enumerate(Y):
        
            for w_pos, x_width_ in enumerate(X):

                # Read again from WSI object wastes tooooo much time.
                # patch_img = wsi_.read_region((x_width_, y_height_), level, (patch_size, patch_size))
                
                patch_arr = wsi_rgb[y_height_: y_height_ + patch_size,\
                                    x_width_:x_width_ + patch_size,:]            
                # print("read_region (scaled coordinates): ", x_width_, y_height_)

                width_mask = x_width_
                height_mask = y_height_                
                
                patch_mask_arr = mask[height_mask: height_mask + patch_size, \
                                      width_mask: width_mask + patch_size]

                # print("Numpy mask shape: ", patch_mask_arr.shape)
                # print("Numpy patch shape: ", patch_arr.shape)

                try:
                    bitwise_ = cv2.bitwise_and(patch_arr, patch_mask_arr)
                
                except Exception as err:
                    print('Out of the boundary')
                    pass
                    
#                     f_ = ((patch_arr > PIXEL_TH) * 1)
#                     f_ = (f_ * PIXEL_WHITE).astype('uint8')
#                     if np.mean(f_) <= (PIXEL_TH + 40):
#                         patches.append(patch_arr)
#                         patches_coords.append((x_width_, y_height_))
#                         print(x_width_, y_height_)
#                         print('Saved\n')

                else:
                    bitwise_grey = cv2.cvtColor(bitwise_, cv2.COLOR_RGB2GRAY)
                    white_pixel_cnt = cv2.countNonZero(bitwise_grey)

                    

                    if white_pixel_cnt >= ((patch_size ** 2) * 0.5):

                        if (patch_arr.shape[0], patch_arr.shape[1])  == \
                        (patch_size, patch_size):

                            patches.append(patch_arr)
                            patches_coords.append((x_width_ + delta_x , 
                                                   y_height_ + delta_y))
                            patches_coords_local.append((x_width_, y_height_))

                            # print("global:", x_width_ + delta_x, y_height_ + delta_y)
                            # print("local: ", x_width_, y_height_)
                            # print('Saved\n')

                    else:
                        pass
                        # print('Did not save\n')

    # end = time.time()
    # print("Time spent on patch extraction: ",  (end - start))

    # patches_ = [patch_[:,:,:3] for patch_ in patches] 
    print("Total number of patches extracted: ", len(patches))
    
    return patches, patches_coords, patches_coords_local

'''
    Parse xml annotation files.
'''
def parse_annotation(anno_path, level, mag_factor):
    
   
   
        
        # print("annotation area #%d", an_i)

        # node_list = list()
        node_list_=list()

        for coor in crds:
            
            x = int(float(coor.attrib['X'])) / mag_factor
            y = int(float(coor.attrib['Y'])) / mag_factor

            x = int(x)
            y = int(y)

            # node_list.append(Point(x,y))
            node_list_.append((x,y))
        
        anno_list.append(node_list_)

        if len(node_list_) > 2:
            polygon_ = Polygon(node_list_)
            polygon_list.append(polygon_)
    
    return polygon_list, anno_list
    

