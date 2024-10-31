import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import csv
from plots import temporal
from utils import image_processing
from utils import rbf_computation
from plots import plot_rbf
from plots import scale_bar
import os

from utils import rbf_computation


# Read M0
M0 = 

# Read kidney mask
# A) GT
kidney_mask = 

# Read previously saved pwi
PWI = 

# Calculte RBF
rbf = rbf_computation.rbf_computation(M0, PWI)

# Mask the rbf using the kidney mask
rbf_masked = rbf * kidney_mask
rbf_image_masked = sitk.GetImageFromArray(rbf_masked)





