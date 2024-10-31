import SimpleITK as sitk
import os
import cv2 as cv
from utils import image_processing
import numpy as np


def calculate_median_img(images): 
    arr = np.array(np.median(images, axis=(0)))
    return arr

def extract_avg_pwi(images, main_path, save=None): 
    images = images.astype(np.float32)
    pwi_serie = image_processing.calculate_pwi_serie(images)
    avg_pwi = image_processing.calculate_mean_img(pwi_serie)
    pwi_path = main_path
    avg_path = main_path + '/avg_Control_Label/'

    m0 = images[0,:,:]
    control_imgs = images[1:]
    control_imgs = control_imgs[1::2]
    label_imgs = images[1:]
    label_imgs = label_imgs[::2]

    avg_control = image_processing.calculate_mean_img(control_imgs)
    avg_label = image_processing.calculate_mean_img(label_imgs)

    substracted_avg = avg_control - avg_label

    avg_control_bym0 = cv.divide(avg_control, m0)
    avg_label_bym0 = cv.divide(avg_label, m0)
    substracted_avg_bym0 = cv.divide(substracted_avg, m0)
    
    if save:
        if not os.path.exists(pwi_path):
            os.makedirs(pwi_path)
        # if not os.path.exists(avg_path):
        #     os.makedirs(avg_path + 'controls/')
        #     os.makedirs(avg_path + 'labels/')
        #     os.makedirs(avg_path + 'substractions/')
        #     os.makedirs(avg_path + 'bym0/controls/')
        #     os.makedirs(avg_path + 'bym0/labels/')
        #     os.makedirs(avg_path + 'bym0/substractions/')

        sitk.WriteImage(sitk.GetImageFromArray(avg_pwi), pwi_path + 'pwi.nii')
        sitk.WriteImage(sitk.GetImageFromArray(avg_control), pwi_path + 'controls.nii')
        sitk.WriteImage(sitk.GetImageFromArray(avg_label), pwi_path + 'labels.nii')
        sitk.WriteImage(sitk.GetImageFromArray(substracted_avg), pwi_path + 'substractions.nii')
        sitk.WriteImage(sitk.GetImageFromArray(avg_control_bym0), pwi_path + 'bym0_controls.nii')
        sitk.WriteImage(sitk.GetImageFromArray(avg_label_bym0), pwi_path + 'bym0_labels.nii')
        sitk.WriteImage(sitk.GetImageFromArray(substracted_avg_bym0), pwi_path + 'bym0_substractions.nii')

        # # If testing one model per study..
        # sitk.WriteImage(sitk.GetImageFromArray(avg_pwi), pwi_path + str(id+1) + '.nii')
        # sitk.WriteImage(sitk.GetImageFromArray(avg_control), avg_path + 'controls/' + str(id+1) + '.nii')
        # sitk.WriteImage(sitk.GetImageFromArray(avg_label), avg_path + 'labels/' + str(id+1) + '.nii')
        # sitk.WriteImage(sitk.GetImageFromArray(substracted_avg), avg_path + 'substractions/' + str(id+1) + '.nii')
        # sitk.WriteImage(sitk.GetImageFromArray(avg_control_bym0), avg_path + 'bym0/controls/' + str(id+1) + '.nii')
        # sitk.WriteImage(sitk.GetImageFromArray(avg_label_bym0), avg_path + 'bym0/labels/' + str(id+1) + '.nii')
        # sitk.WriteImage(sitk.GetImageFromArray(substracted_avg_bym0), avg_path + 'bym0/substractions/' + str(id+1) + '.nii')

    return avg_pwi
    
    # elif image_group == 'Tested_Controls':
    #     avg_path = main_path + '/Voxelmorph/Native/Results/' + studies[nstudies] + main_modelname + '/' + loss_opt + '/' + experiment + '/' + image_group 
    #     avg_control = image_processing.calculate_mean_img(images)
    #     if not os.path.exists(avg_path + '/avg_controls/'):
    #         os.makedirs(avg_path + '/avg_controls/')
    #     sitk.WriteImage(sitk.GetImageFromArray(avg_control), avg_path + '/avg_controls/' + str(id+1) + '.nii')

    # elif image_group == 'Tested_Labels':
    #     avg_path = main_path + '/Voxelmorph/Native/Results/' + studies[nstudies] + main_modelname + '/' + loss_opt + '/' + experiment + '/' + image_group 
    #     avg_label = image_processing.calculate_mean_img(images)
    #     if not os.path.exists(avg_path + '/avg_labels/'):
    #         os.makedirs(avg_path + '/avg_labels')
    #     sitk.WriteImage(sitk.GetImageFromArray(avg_label), avg_path + '/avg_labels/' + str(id+1) + '.nii')


 

def compute_mean(img, mask):
    # Threshold the mask (convert values to 0 or 255)
    _, mask = cv.threshold(mask, 0.5, 255, cv.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    masked_image = cv.bitwise_and(img, img, mask = mask)
    # Convert masked_image to float type to handle NaNs
    masked_image = masked_image.astype(np.float32)
  # Set masked pixels with value 0 to NaN
    masked_image[masked_image == 0] = np.nan

    return np.nanmean(masked_image.reshape(-1))

def compute_tsnr(mean_values): 
    tsnr_value = np.nanmean(mean_values)/np.nanstd(mean_values)
    
    return tsnr_value

def erode_mask(mask_to_erode, kernel_size=2):
    # Define the structuring element (kernel) for erosion
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform erosion
    eroded_mask = cv.erode(mask_to_erode, kernel, iterations=1)
    return eroded_mask

def calculate_pwi_serie(img_serie, mode = None): 
    # img_serie.astype(np.int16)
    if (img_serie.shape[0] % 2) == 0: # es par
        control_idx = range(1, img_serie.shape[0], 2)
        label_idx = range(0, img_serie.shape[0], 2)
    else:
        control_idx = range(2, img_serie.shape[0], 2)
        label_idx = range(1, img_serie.shape[0], 2)
    
    pwi_serie = sitk.Image([img_serie.shape[2], img_serie.shape[1], len(control_idx)], sitk.sitkFloat32)
    pwi_serie_arr = sitk.GetArrayFromImage(pwi_serie)
    for pairs in range(0, len(control_idx)):
        if mode == "inverted": 
            pwi_serie_arr[pairs,:,:] = img_serie[label_idx[pairs],:,:] - img_serie[control_idx[pairs],:,:]
        else:
            pwi_serie_arr[pairs,:,:] = img_serie[control_idx[pairs],:,:] - img_serie[label_idx[pairs],:,:]

    return pwi_serie_arr


def ASL_processing_allograft(images, cortexMaskMedian, filter=True):
   # Extract pwi series or perfusion maps
    PWIs_PRE = image_processing.calculate_pwi_serie(images)
    control_idx = range(2, images.shape[0], 2)
    label_idx = range(1, images.shape[0], 2)

    # Erode masks
    eroded_mask = image_processing.erode_mask(cortexMaskMedian, 2)

    # PRE --→ Intensity values list
    corticalAU_PRE = []

    for pwi_imgs in range(0, PWIs_PRE.shape[0]): 
        corticalAU_PRE.append(image_processing.compute_mean(images[control_idx[pwi_imgs], :, :], eroded_mask) - image_processing.compute_mean(images[label_idx[pwi_imgs], :, :], eroded_mask))
        
     # PRE --→ Intensity values list
    if filter:
        pos_threshold = np.nanmean(corticalAU_PRE) + (2 * np.nanstd(corticalAU_PRE))
        neg_threshold = np.nanmean(corticalAU_PRE) - (2 * np.nanstd(corticalAU_PRE))

        corticalAU_POST = corticalAU_PRE.copy()

        PWIs_POST = PWIs_PRE.copy()
        
        for i in range(0, PWIs_PRE.shape[0]): 
            if not neg_threshold < corticalAU_PRE[i] < pos_threshold:
                corticalAU_POST[i] = np.nan
                PWIs_POST[i,:,:] = np.nan
            
        tsnr = np.nanmean(corticalAU_POST)/np.nanstd(corticalAU_POST)
        
        Averaged_PWIs_PRE = np.nanmean(PWIs_PRE, axis=0)
        
        return  corticalAU_PRE, corticalAU_POST, \
                PWIs_PRE, PWIs_POST,\
                Averaged_PWIs_PRE, \
                tsnr
    
def ASL_processing_native(images, cortexMaskMedian_R, cortexMaskMedian_L, filter=True):
   # Extract pwi series or perfusion maps
    PWIs_PRE = image_processing.calculate_pwi_serie(images)
    control_idx = range(2, images.shape[0], 2)
    label_idx = range(1, images.shape[0], 2)

    # Erode masks
    eroded_mask_L = image_processing.erode_mask(cortexMaskMedian_L, 2)
    eroded_mask_R = image_processing.erode_mask(cortexMaskMedian_R, 2)

    # PRE --→ Intensity values list
    corticalAU_Right_PRE = []
    corticalAU_Left_PRE = []

    for pwi_imgs in range(0, PWIs_PRE.shape[0]): 
        corticalAU_Right_PRE.append(image_processing.compute_mean(images[control_idx[pwi_imgs], :, :], eroded_mask_R) - image_processing.compute_mean(images[label_idx[pwi_imgs], :, :], eroded_mask_R))
        corticalAU_Left_PRE.append(image_processing.compute_mean(images[control_idx[pwi_imgs], :, :], eroded_mask_L) - image_processing.compute_mean(images[label_idx[pwi_imgs], :, :], eroded_mask_L))
        
     # PRE --→ Intensity values list
    if filter:
        pos_Left_threshold = np.nanmean(corticalAU_Left_PRE) + (2 * np.nanstd(corticalAU_Left_PRE))
        neg_Left_threshold = np.nanmean(corticalAU_Left_PRE) - (2 * np.nanstd(corticalAU_Left_PRE))
        pos_Right_threshold = np.nanmean(corticalAU_Right_PRE) + (2 * np.nanstd(corticalAU_Right_PRE))
        neg_Right_threshold = np.nanmean(corticalAU_Right_PRE) - (2 * np.nanstd(corticalAU_Right_PRE))

        corticalAU_Left_POST = corticalAU_Left_PRE.copy()
        corticalAU_Right_POST = corticalAU_Right_PRE.copy()

        PWIs_Right_POST = PWIs_PRE.copy()
        PWIs_Left_POST = PWIs_PRE.copy()
        
        for i in range(0, PWIs_PRE.shape[0]): 
            if not neg_Left_threshold < corticalAU_Left_PRE[i] < pos_Left_threshold:
                corticalAU_Left_POST[i] = np.nan
                PWIs_Left_POST[i,:,:] = np.nan
            if not neg_Right_threshold < corticalAU_Right_PRE[i] < pos_Right_threshold:
                corticalAU_Right_POST[i] = np.nan
                PWIs_Right_POST[i,:,:] = np.nan
        
        tsnr_right = np.nanmean(corticalAU_Right_POST)/np.nanstd(corticalAU_Right_POST)
        tsnr_left = np.nanmean(corticalAU_Left_POST)/np.nanstd(corticalAU_Left_POST)
        
        Averaged_PWIs_PRE = np.nanmean(PWIs_PRE, axis=0)
        Averaged_PWIs_Right_POST = np.nanmean(PWIs_Right_POST, axis=0)
        Averaged_PWIs_Left_POST = np.nanmean(PWIs_Left_POST, axis=0)
        
        return  corticalAU_Right_PRE, corticalAU_Right_POST, \
                corticalAU_Left_PRE, corticalAU_Left_POST, \
                PWIs_PRE, PWIs_Right_POST, PWIs_Left_POST, \
                Averaged_PWIs_PRE, Averaged_PWIs_Right_POST, Averaged_PWIs_Left_POST, \
                tsnr_right, tsnr_left

def orientation_detection(img): 
    # Apply a Gaussian blur to the image to remove noise
    gray = cv.GaussianBlur((img*255).astype(np.uint8), (5, 5), 0)
    # Apply Canny edge detection to detect edges in the image
    edges = cv.Canny((gray*255).astype(np.uint8), 50, 100)

    # Find contours in the image
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)

    # Iterate through each contour and determine whether it corresponds to a right or left mask
    for contour in contours:
        # Compute the bounding box of the contour
        x, y, w, h = cv.boundingRect(contour)
        
        # Determine the center of the bounding box
        cx = x + w // 2
        cy = y + h // 2
        
        # If the center of the bounding box is on the left side of the image, it is a left mask; otherwise, it is a right mask
        # Right
        if cx < img.shape[1] // 2:
            mask[y:y+h, x:x+w] = 1
        # Left
        else:
            mask[y:y+h, x:x+w] = 2
    return mask

def label_right_left(img): 
    '''
    It receives a sitk format image and returns an array with relabeled mask. Right mask = label 1. Left mask = label 2. 
    '''
    mask_right_total = np.zeros_like(img)
    mask_left_total = np.zeros_like(img)

    if len(img.shape) == 2: # 2d image
        img_unique = img
        mask = orientation_detection(img_unique)

        # Relabel original mask
        mask_right = np.logical_and(img_unique > 0, mask == 1)
        mask_left = np.logical_and(img_unique > 0, mask == 2)

        mask_right_total = mask_right
        mask_left_total = mask_left


    else:
        for i in range(0, img.shape[0]): 
            img_unique= img[i,:,:]
            mask = orientation_detection(img_unique)

            # Relabel original mask
            mask_right = np.logical_and(img_unique > 0, mask == 1)
            mask_left = np.logical_and(img_unique > 0, mask == 2)

            # mask_total[i, :, :] = mask_dual
            mask_right_total[i:] = mask_right
            mask_left_total[i:] = mask_left

    return mask_right_total.astype(np.uint8), mask_left_total.astype(np.uint8)

def fill_holes(label):
    '''
    It fills the holes (for example, for abdominal aorta, it fills the lumen hole that appears in the mask)
    :param label: label where the holes will be removed (simple itk)
    :return: label with removed holes
    '''

    fillHole=sitk.GrayscaleFillholeImageFilter

    if label.GetDimension()==3:
        x, y, z= label.GetSize()
        np_label = sitk.GetArrayFromImage(label)

        for slice in range(z):

            #para asignar un conjunto de filas y columnas hay que utilizar numpy
            noHole = sitk.GrayscaleFillhole(label[:, :, slice])

            np_label[slice, :, :] = sitk.GetArrayFromImage(noHole)

    elif label.GetDimension()==2:
        noHole = sitk.GrayscaleFillhole(label)
        np_label = sitk.GetArrayFromImage(noHole)

    withoutHoles=sitk.GetImageFromArray(np_label)
    withoutHoles.SetOrigin(label.GetOrigin())
    withoutHoles.SetSpacing(label.GetSpacing())

    return withoutHoles