
import numpy as np
import SimpleITK as sitk
from utils import image_processing
import matplotlib.pyplot as plt


def rbf_computation(M0, PWI): 
    '''
    PWIs_Right_POST lo hemos elegido como PWI (considerando que right y left es igual)
    '''
    lambda_val = 0.9
    delay = 0.058   
    pld = 1.200
    t1b = 1.650 
    alfa = 0.74 * 0.93 * 0.93 #%antes con 0.75
    tau = 1.600
   
    # RBF computation
    rbf = np.zeros((96, 96), dtype=np.float64)
    rbf= (6000 * lambda_val * PWI * np.exp(pld / t1b)) / (2 * alfa * t1b * M0 * (1 - np.exp(-tau / t1b)))
    rbf[rbf == 0] = np.nan

    # Delete very high values
    rbf_thres = rbf < 1000
    rbf_final = rbf * rbf_thres


    return rbf_final

def calculate_rbf(asl_serie, cortex=None, medulla=None, subject=None): 
    asl_serie = sitk.GetArrayFromImage(asl_serie)
    # Cortex
    cortex_masks = sitk.GetArrayFromImage(cortex)
    cortex_mask_median = None  # Inicializar la variable


    if len(cortex_masks.shape) > 2:  # Si hay más de una imagen
        if subject == 'HK': 
            r_cortex_masks, l_cortex_masks = image_processing.label_right_left(cortex_masks)
            cortexMaskMedianR = image_processing.calculate_median_img(r_cortex_masks[1:])
            cortexMaskMedianR[cortexMaskMedianR >= 1] = 1
            cortexMaskMedianR[cortexMaskMedianR < 1] = 0
            
            cortexMaskMedianL = image_processing.calculate_median_img(l_cortex_masks[1:])
            cortexMaskMedianL[cortexMaskMedianL >= 1] = 1
            cortexMaskMedianL[cortexMaskMedianL < 1] = 0
            
            cortex_mask_median = (cortexMaskMedianR, cortexMaskMedianL)  # Guarda ambas medianas

        elif subject == 'TK':
            cortexMaskMedian = image_processing.calculate_median_img(cortex_masks[1:])
            cortexMaskMedian[cortexMaskMedian >= 1] = 1
            cortexMaskMedian[cortexMaskMedian < 1] = 0
            cortex_mask_median = cortexMaskMedian  # Guarda la mediana
    
    else:  # Si no hay más de una imagen, usa la máscara original
        if subject == 'HK':
            cortexMaskMedianR, cortexMaskMedianL = image_processing.label_right_left(cortex_masks)
            # plt.imshow(cortexMaskMedianR)
            # plt.show()
            # plt.imshow(cortexMaskMedianL)
            # plt.show()
            cortex_mask_median = (cortexMaskMedianR, cortexMaskMedianL)
        elif subject == 'TK': 
            cortex_mask_median = cortex_masks

    if medulla: 
        medulla_masks = sitk.GetArrayFromImage(medulla)
        medulla_mask_median = None  # Inicializar la variable

        if len(medulla_masks.shape) > 2: # Si hay más de una imagen
            if subject == 'HK': 
                r_medulla_masks, l_medulla_masks = image_processing.label_right_left(medulla_masks)
                medullaMaskMedianR = image_processing.calculate_median_img(r_medulla_masks[1:])
                medullaMaskMedianR[medullaMaskMedianR >= 1] = 1
                medullaMaskMedianR[medullaMaskMedianR < 1] = 0
                
                medullaMaskMedianL = image_processing.calculate_median_img(l_medulla_masks[1:])
                medullaMaskMedianL[medullaMaskMedianL >= 1] = 1
                medullaMaskMedianL[medullaMaskMedianL < 1] = 0
                
                medulla_mask_median = (medullaMaskMedianR, medullaMaskMedianL)  # Guarda ambas medianas

            elif subject == 'TK':
                medullaMaskMedian = image_processing.calculate_median_img(medulla_masks[1:])
                medullaMaskMedian[medullaMaskMedian >= 1] = 1
                medullaMaskMedian[medullaMaskMedian < 1] = 0
                medullaMaskMedian = medullaMaskMedian  # Guarda la mediana
        
        else:  # Si no hay más de una imagen, usa la máscara original
            if subject == 'HK':
                medullaMaskMedianR, medullaMaskMedianL = image_processing.label_right_left(medulla_masks)
                medulla_mask_median = (medullaMaskMedianR, medullaMaskMedianL)
            elif subject == 'TK': 
                medulla_mask_median = medulla_masks

    if subject == 'HK': 
        # Extract RBF values
        (corticalAU_Right_PRE, corticalAU_Right_POST, 
        corticalAU_Left_PRE, corticalAU_Left_POST, 
        PWIs_PRE, PWIs_Right_POST, PWIs_Left_POST, 
        Averaged_PWIs_PRE, Averaged_PWIs_Right_POST, Averaged_PWIs_Left_POST,
        tsnr_right, tsnr_left) = image_processing.ASL_processing_native(asl_serie, cortex_mask_median[0], cortex_mask_median[1])
        
        M0 = asl_serie[0, :, :]
        rbf = rbf_computation(M0, Averaged_PWIs_Right_POST)
        rbf = np.where((rbf > 1000) | (rbf < -1000), np.nan, rbf)
        rbf_image = rbf

        plt.imshow(rbf)

        RightCortex_rbf = image_processing.compute_mean(rbf, cortex_mask_median[0])
        LeftCortex_rbf = image_processing.compute_mean(rbf, cortex_mask_median[1])

        if medulla:
            RightMedulla_rbf = image_processing.compute_mean(rbf, medulla_mask_median[0])
            LeftMedulla_rbf = image_processing.compute_mean(rbf, medulla_mask_median[1])
            
            return Averaged_PWIs_Right_POST, RightCortex_rbf, LeftCortex_rbf, RightMedulla_rbf, LeftMedulla_rbf, rbf_image
        else: 
            return Averaged_PWIs_Right_POST, RightCortex_rbf, LeftCortex_rbf, rbf_image
    
    elif subject == 'TK': 
        # Extract RBF values
        (corticalAU_PRE, 
        corticalAU_PRE, 
        PWIs_PRE, 
        Averaged_PWIs_PRE, Averaged_PWIs_POST,
        tsnr) = image_processing.ASL_processing_allograft(asl_serie, cortex_mask_median)
        
        M0 = asl_serie[0, :, :]
        rbf = rbf_computation(M0, Averaged_PWIs_POST)
        rbf_image = rbf

        # plt.imshow(Averaged_PWIs_POST)
        # plt.show()

        rbf_cortex = image_processing.compute_mean(rbf, cortex_mask_median)

        if medulla:
            rbf_medulla = image_processing.compute_mean(rbf, medulla_mask_median)
            return Averaged_PWIs_POST, rbf_cortex, rbf_medulla, rbf_image

        else:
            return Averaged_PWIs_POST, rbf_cortex, rbf_image

