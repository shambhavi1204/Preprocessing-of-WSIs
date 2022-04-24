fimport numpy as np
import openslide
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image
import os

img_path = '/home/shambhavi/Downloads/tumor_002.tif'

#os.mkdir('image_1')

#method 1 
slide1 = openslide.OpenSlide (img_path)
#print(slide1.properties)
level = 3
mag_factor = pow(2, level)

#orig_size = slide1.shape[:2]
#print(orig_size)
slide_w_, slide_h_ = slide1.level_dimensions[level]

print(slide_h_)
print(slide_w_)

rgba_image_pil = slide1.read_region((0, 0), level, (slide_w_, slide_h_))
#print("width, height:", rgba_image_pil.size)

rgba_image = np.asarray(rgba_image_pil)
#print("transformed:", rgba_image.shape)

r_, g_, b_, a_ = cv2.split(rgba_image)
    
wsi_rgb_ = cv2.merge((r_, g_, b_))
    
wsi_gray_ = cv2.cvtColor(wsi_rgb_,cv2.COLOR_RGB2GRAY)
wsi_hsv_ = cv2.cvtColor(wsi_rgb_, cv2.COLOR_RGB2HSV)

#print(wsi_rgb_.shape)
#print(wsi_rgb_.dtype)

orig_size = wsi_rgb_.shape[:2]

print(orig_size)

out_size = (224,224)
num_crops = [math.ceil(orig_size[i] / out_size[i]) for i in range(2)]

crop_offset = []
for i in range(2):
    if num_crops[i] > 1:
        crop_offset.append(int(np.floor((num_crops[i] * out_size[i] - orig_size[i]) / (num_crops[i] - 1))))
    else:
        crop_offset.append(0)

total_crops = np.prod(np.array(num_crops))
patches = np.zeros((total_crops, out_size[0], out_size[1], 3))
iter_patch = 0
for iter_row in range(num_crops[0]):
    if iter_row < num_crops[0] - 1:
        start_i = iter_row * (out_size[0] - crop_offset[0])
    else:
        start_i = orig_size[0] - out_size[0]
    end_i = start_i + out_size[0]
    for iter_col in range(num_crops[1]):
        if iter_col < num_crops[1] - 1:
            start_j = iter_col * (out_size[1] - crop_offset[1])
        else:
            start_j = orig_size[1] - out_size[1]
        end_j = start_j + out_size[1]
        patches[iter_patch] = wsi_rgb_[start_i:end_i, start_j:end_j, :]
        iter_patch += 1
        #fig,(ax) = plt.subplots(figsize=(8,4), ncols=1)
       # ax.imshow(patches[iter_patch][0])
        #fig.tight_layout()
        #plt.show()
       # break

print(patches.shape)
print(patches[3555])
print(patches[3700].shape)
cv2.imshow('frame',patches[3555])
cv2.waitKey(0) 
cv2.destroyAllWindows()
#plt.show()
        
