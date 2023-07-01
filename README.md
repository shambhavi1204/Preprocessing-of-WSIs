Patches extracted with 25 percent overlap.


converted to hsv (0.65>hue>0.5, saturation>0.1, 0.9> value > 0.5)

use of contrast filter ,patches with more than 25 percent of foreground to background ratio is choosen.

use of affine transformation and random cropping(preprocessing.py)

Use of color jitter rather than strain normalization as it gives better results.(preprocessing.py)

Vectorize.py -> The provided code loads an image file using the OpenSlide library, extracts a region of interest (ROI) from the image at a specified level, and performs various image processing operations on the ROI. It splits the RGBA image into separate channels, merges the RGB channels, converts the RGB image to grayscale and HSV color space. It then divides the image into smaller patches of a specified size and stores them in a NumPy array. Finally, it displays one of the patches using OpenCV's cv2.imshow function. The code is useful for performing preprocessing steps on whole-slide images for tasks such as image segmentation or analysis in the field of digital pathology.
