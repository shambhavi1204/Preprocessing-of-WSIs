Patches extracted with 25 percent overlap.


converted to hsv (0.65>hue>0.5, saturation>0.1, 0.9> value > 0.5)

use of contrast filter ,patches with more than 25 percent of foreground to background ratio is choosen.

use of affine transformation and random cropping(preprocessing.py)

Use of color jitter rather than strain normalization as it gives better results.(preprocessing.py)

Vectorize.py -> The provided code loads an image file using the OpenSlide library, extracts a region of interest (ROI) from the image at a specified level, and performs various image processing operations on the ROI. It splits the RGBA image into separate channels, merges the RGB channels, converts the RGB image to grayscale and HSV color space. It then divides the image into smaller patches of a specified size and stores them in a NumPy array. Finally, it displays one of the patches using OpenCV's cv2.imshow function. The code is useful for performing preprocessing steps on whole-slide images for tasks such as image segmentation or analysis in the field of digital pathology.

Utils.py --->

The provided code contains various functions for processing whole-slide images (WSIs) and their corresponding annotations. It imports necessary libraries and defines some constants such as patch size, channel number, class number, etc. The openSlide_init function initializes an OpenSlide object for the WSI file specified by the file path and returns it. The read_wsi function reads a region of interest (ROI) from the WSI at a given level and returns it as a NumPy array. The construct_colored_wsi function splits the RGBA image into separate channels (R, G, B, and A), merges the RGB channels, and converts the RGB image to grayscale and HSV color space. The get_contours function extracts contours from a binary image and returns the bounding boxes, contour coordinates, contours, and a mask image.


The segmentation_hsv function performs segmentation on the HSV image using a specified threshold range and returns the resulting contours and mask image. The construct_bags function extracts patches from the WSI based on the contours and mask image, and returns the patches, their coordinates, and local coordinates. The parse_annotation function parses XML annotation files and returns the polygon coordinates and annotation lists. These functions can be used for tasks such as WSI analysis and annotation processing in the field of digital pathology.
