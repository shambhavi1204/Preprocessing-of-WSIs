Patches extracted with 25 percent overlap.


converted to hsv (0.65>hue>0.5, saturation>0.1, 0.9> value > 0.5)

use of contrast filter ,patches with more than 25 percent of foreground to background ratio is choosen.

use of affine transformation and random cropping(preprocessing.py)

Use of color jitter rather than strain normalization as it gives better results.(preprocessing.py)

extract_patches.py: includes processing functions for WSIs of which level >= 3;

extract_patches_split.py: includes processing functions for WSIs of which level <= 2.
