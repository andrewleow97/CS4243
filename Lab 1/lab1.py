import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import math
# You are free to use NumPy functions, but you may not use numpy.pad()or other built-in functions of OpenCV aside from those already in the code template.
##### Part 1: image preprossessing #####

def rgb2gray(img):
    """
    5 points
    Convert a colour image greyscale
    Use (R,G,B)=(0.299, 0.587, 0.114) as the weights for red, green and blue channels respectively
    :param img: numpy.ndarray (dtype: np.uint8)
    :return gray_image: numpy.ndarray (dtype:np.uint8)
    """
    if len(img.shape) != 3:
        print('RGB Image should have 3 channels')
        return
    imgH, imgW, _ = img.shape
    img_gray = np.zeros([imgH, imgW], dtype=np.uint8)
    for i in range(imgH):
        for j in range(imgW):
            img_gray[i][j] = (img[i][j][0]*0.299 + img[i][j][1]*0.587 + img[i][j][2]*0.114).astype(np.uint8)  
    
    # can also use matrix multiplication via dot product with np.dot on the weights
    return img_gray

def convolve(img, kernel):
    # for filter of size (2k+1, 2k+1), need to pad k zeroes around img
    imgH, imgW = img.shape[:2]
    kH, kW = kernel.shape[:2]
    pad = (kW-1) // 2 
    img_padded = pad_zeros(img, pad, pad, pad, pad) #padding
    kernel = np.flip(kernel) # flipping kernel for convolution instead of cross correlation
    output = np.zeros((imgH, imgW), dtype=float) # creating output matrix
    for y in np.arange(pad, imgH+pad):
        for x in np.arange(pad, imgW+pad):
            roi = img_padded[y-pad:y+pad+1, x-pad:x+pad+1].astype(float) # get roi around pixel
            k = float(np.vdot(roi, kernel)) # convolving roi and kernel
            output[y-pad, x-pad] = k
    return output


def gray2grad(img):
    """
    5 points
    Estimate the gradient map from the grayscale images by convolving with Sobel filters (horizontal and vertical gradients) and Sobel-like filters (gradients oriented at 45 and 135 degrees)
    The coefficients of Sobel filters are provided in the code below.
    :param img: numpy.ndarray
    :return img_grad_h: horizontal gradient map. numpy.ndarray
    :return img_grad_v: vertical gradient map. numpy.ndarray
    :return img_grad_d1: diagonal gradient map 1. numpy.ndarray
    :return img_grad_d2: diagonal gradient map 2. numpy.ndarray
    """
    sobelh = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]], dtype = float)
    sobelv = np.array([[-1, -2, -1], 
                       [0, 0, 0], 
                       [1, 2, 1]], dtype = float)
    sobeld1 = np.array([[-2, -1, 0],
                        [-1, 0, 1],
                        [0,  1, 2]], dtype = float)
    sobeld2 = np.array([[0, -1, -2],
                        [1, 0, -1],
                        [2, 1, 0]], dtype = float)
    
    img_grad_h = convolve(img, sobelh)
    img_grad_v = convolve(img, sobelv)
    img_grad_d1 = convolve(img, sobeld1)
    img_grad_d2 = convolve(img, sobeld2)       
    
    return img_grad_h, img_grad_v, img_grad_d1, img_grad_d2

def pad_zeros(img, pad_height_bef, pad_height_aft, pad_width_bef, pad_width_aft):
    """
    5 points
    Add a border of zeros around the input images so that the output size will match the input size after a convolution or cross-correlation operation.
    e.g., given matrix [[1]] with pad_height_bef=1, pad_height_aft=2, pad_width_bef=3 and pad_width_aft=4, obtains:
    [[0 0 0 0 0 0 0 0]
    [0 0 0 1 0 0 0 0]
    [0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0]]
    :param img: numpy.ndarray
    :param pad_height_bef: int
    :param pad_height_aft: int
    :param pad_width_bef: int
    :param pad_width_aft: int
    :return img_pad: numpy.ndarray. dtype is the same as the input img. 
    """
    height, width = img.shape[:2]
    new_height, new_width = (height + pad_height_bef + pad_height_aft), (width + pad_width_bef + pad_width_aft)
    if len(img.shape) == 2:
        img_pad = np.zeros((new_height, new_width), dtype=img.dtype)
        img_pad[pad_height_bef:-pad_height_aft, pad_width_bef:-pad_width_aft] = img
    else:
        img_pad = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)
        img_pad[pad_height_bef:-pad_height_aft, pad_width_bef:-pad_width_aft, :] = img
        
    return img_pad


##### Part 2: Normalized Cross Correlation #####
def normalized_cross_correlation(img, template):
    """
    10 points.
    Implement the convolution operation in a naive 6 nested for-loops. 
    The 6 loops include the height, width, channel of the output and height, width and channel of the template.
    :param img: numpy.ndarray.
    :param template: numpy.ndarray.
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1
    response = np.zeros((Ho, Wo), np.float)
    pad_height_bef, pad_height_aft = template.shape[0] // 2 - (1 if template.shape[0] % 2 == 0 else 0), template.shape[0] // 2
    pad_width_bef, pad_width_aft = template.shape[1] // 2 - (1 if template.shape[1] % 2 == 0 else 0), template.shape[1] // 2
    
    # this is |F|, constant across all loops
    norm_kernel = np.sum(np.square(template, dtype=float))

    for i_height in range(pad_height_bef, Hi-pad_height_aft):
        for i_width in range(pad_width_bef, Wi-pad_width_aft):
            # HoxWo in middle of HixWi, must revert to HoxWo indices
            out_height = i_height - pad_height_bef
            out_width = i_width - pad_width_bef
            corr_sum = 0.0
            w_sq = 0.0
            #multiplying each pixel in the template by the image pixel that it overlaps and then summing the results over all the pixels of the template
            for t_height in range(Hk):
                for t_width in range(Wk):
                    if len(img.shape) == 3: # RGB image and template
                        for t_channel in range(template.shape[2]):
                            #sum of fuvc * pi+u, j+v, c
                            corr_sum += np.multiply(img[i_height + t_height - pad_height_bef][i_width + t_width - pad_width_bef][t_channel], template[t_height][t_width][t_channel], dtype=float)
                            #part of |wij|
                            w_sq += np.square(img[i_height + t_height - pad_height_bef][i_width + t_width - pad_width_bef][t_channel], dtype=float)
                    else: #grayscale image and template
                        corr_sum += np.multiply(img[i_height + t_height - pad_height_bef][i_width + t_width - pad_width_bef], template[t_height][t_width], dtype=float)
                        w_sq += np.square(img[i_height + t_height - pad_height_bef][i_width + t_width - pad_width_bef], dtype=float)
            # sum for xij here = (fu,v,c * pi+u, j+v, c)/|F||wij| 
            response[out_height][out_width] = float(corr_sum/np.sqrt(norm_kernel*(w_sq)))
    
    return response


def normalized_cross_correlation_fast(img, template):
    """
    10 points.
    Implement the cross correlation with 3 nested for-loops. 
    The for-loop over the template is replaced with the element-wise multiplication between the kernel and the image regions.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1
    response = np.zeros((Ho, Wo), np.float)
    pad_height_bef, pad_height_aft = template.shape[0] // 2 - (1 if template.shape[0] % 2 == 0 else 0), template.shape[0] // 2
    pad_width_bef, pad_width_aft = template.shape[1] // 2 - (1 if template.shape[1] % 2 == 0 else 0), template.shape[1] // 2
    norm_kernel = np.sum(np.square(template, dtype=float))
    for i_height in range(pad_height_bef, Hi-pad_height_aft):
        for i_width in range(pad_width_bef, Wi-pad_width_aft):
            out_height = i_height - pad_height_bef
            out_width = i_width - pad_width_bef
            corr_sum = 0.0
            w_sq = 0.0
            if len(img.shape) == 3:
                for channel in range(img.shape[2]):
                    roi = img[i_height - pad_height_bef:i_height+pad_height_aft+1, i_width - pad_width_bef:i_width + pad_width_aft+1, channel]
                    corr_sum += np.sum(np.multiply(roi, template[:,:,channel], dtype=float))
                    w_sq += np.sum(np.square(roi, dtype=float))
            else:
                roi = img[i_height - pad_height_bef:i_height+pad_height_aft+1, i_width - pad_width_bef:i_width + pad_width_aft+1]
                corr_sum += np.sum(np.multiply(roi, template, dtype=float))
                w_sq += np.sum(np.square(roi, dtype=float))
            response[out_height][out_width] = float(corr_sum/np.sqrt(norm_kernel*(w_sq)))
    return response


def normalized_cross_correlation_matrix(img, template):
    """
    10 points.
    Converts cross-correlation into a matrix multiplication operation to leverage optimized matrix operations.
    Please check the detailed instructions in the pdf file.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    # Useful functions : np.stack((a1,a2,a3),axis=1)
    #template.ravel().reshape(3*Xk*Yk,1) for making the 3*Hk*Wk , 1 matrix
    #https://towardsdatascience.com/reshaping-numpy-arrays-in-python-a-step-by-step-pictorial-tutorial-aed5f471cf0b
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1
    response = []
    Fr = []
    Pr = []
    norm_kernel = np.sum(np.square(template, dtype=float))
    
    if len(template.shape) == 3:
        Fr.append(template[:,:,0].flatten())
        Fr.append(template[:,:,1].flatten())
        Fr.append(template[:,:,2].flatten())
    else:
        Fr.append(template[:,:].flatten())
    # Fr has dimension HfWf x 1 = (3*HkWk, 1)
    Fr = np.array(Fr).flatten().reshape(3*Hk*Wk,1)
 
    norm_factor = np.empty((Ho,Wo))
    for x in range(Ho):
        for y in range(Wo): # for region of output
            temp = []
            w_sq = 0.0
            if len(img.shape) == 3:
                # for channel in range(img.shape[2]):
                #     roi = img[x:x+Hk,y:y+Wk,channel].astype(float)
                #     temp.append(roi.flatten())
                #     w_sq += np.sum(np.square(roi))
                temp.append(img[x:x+Hk,y:y+Wk,0].astype(float))
                temp.append(img[x:x+Hk,y:y+Wk,1].astype(float))
                temp.append(img[x:x+Hk,y:y+Wk,2].astype(float))
                w_sq += np.sum(np.square(np.array(temp))).astype(float)
            else:
                roi = img[x:x+Hk,y:y+Wk].astype(float)
                temp.append(roi.flatten())
                w_sq += np.sum(np.square(roi)).astype(float)
            temp = np.array(temp)
            temp = temp.flatten().reshape(-1) # Remaking temp such that its shape is (1,12), instead of 3 lists
            norm_factor[x][y] = 1/(np.sqrt(norm_kernel*w_sq))
            Pr.append(temp) # append 3*Hk*Wk long list into Pr
    
    Pr = np.stack(Pr) # create Ho*Wo rows from the lists in Pr, outputting a (Ho*Wo, 3*Hk*Wk) matrix
    response = np.dot(Pr,Fr) # perform matrix multiplication for cross correlation output


    response = response.reshape(Ho, Wo) # reshape output back into Ho*Wo
    for x in range(Ho):
        for y in range(Wo): # for region of output
            response[x][y] *= norm_factor[x][y] # apply norm factor for each response
    return response


##### Part 3: Non-maximum Suppression #####

def non_max_suppression(response, suppress_range, threshold=None):
    """
    10 points
    Implement the non-maximum suppression for translation symmetry detection
    The general approach for non-maximum suppression is as follows:
	1. Set a threshold τ; values in X<τ will not be considered.  Set X<τ to 0.  
    2. While there are non-zero values in X
        a. Find the global maximum in X and record the coordinates as a local maximum.
        b. Set a small window of size w×w points centered on the found maximum to 0.
	3. Return all recorded coordinates as the local maximum.
    :param response: numpy.ndarray, output from the normalized cross correlation
    :param suppress_range: a tuple of two ints (H_range, W_range). 
                           the points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    :param threshold: int, points with value less than the threshold are set to 0
    :return res: a sparse response map which has the same shape as response
    """
    height, width = suppress_range
    res = np.zeros_like(response, dtype=float)
    if threshold != None:
        for i in range(response.shape[0]):
            for j in range(response.shape[1]):
                if response[i][j] < threshold:
                    response[i][j] = 0 # Set X<τ = 0
    while np.amax(response) > 0:
        max = np.amax(response)
        coords = np.where(response == max)
        i = coords[0][0]
        j = coords[1][0]
        res[i][j] = response[i][j]
        response[i][j] = 0
        for h in range(-height, height):
            for w in range(-width, width):
                if (i+h >= 0) and (i+h < response.shape[0]) and (j+w >= 0) and (j+w < response.shape[1]): # boundary check
                    response[i+h][j+w] = 0
    return res

##### Part 4: Question And Answer #####
    
def normalized_cross_correlation_ms(img, template):
    """
    10 points
    Please implement mean-subtracted cross correlation which corresponds to OpenCV TM_CCOEFF_NORMED.
    For simplicty, use the "fast" version.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1
    response = np.zeros((Ho, Wo), dtype=float)
    pad_height_bef, pad_height_aft = template.shape[0] // 2 - (1 if template.shape[0] % 2 == 0 else 0), template.shape[0] // 2
    pad_width_bef, pad_width_aft = template.shape[1] // 2 - (1 if template.shape[1] % 2 == 0 else 0), template.shape[1] // 2
    template_mean = (template.sum()/np.square((2*pad_height_bef)+1))
    template = np.subtract(template,template_mean, dtype=float)
    # print(template)
    norm_kernel = np.sum(np.square(template, dtype=float))
    for i_height in range(pad_height_bef, Hi-pad_height_aft):
        for i_width in range(pad_width_bef, Wi-pad_width_aft):
            out_height = i_height - pad_height_bef
            out_width = i_width - pad_width_bef
            corr_sum = 0.0
            w_sq = 0.0
            if len(img.shape) == 3:
                for channel in range(img.shape[2]):
                    roi = img[i_height - pad_height_bef:i_height+pad_height_aft+1, i_width - pad_width_bef:i_width + pad_width_aft+1, channel]
                    roi_mean = roi.sum()/np.square((2*pad_height_bef)+1).astype(float)
                    roi = np.subtract(roi,roi_mean, dtype=float)
                    corr_sum += np.sum(np.multiply(roi, template[:,:,channel], dtype=float))
                    w_sq += np.sum(np.square(roi, dtype=float))
            else:
                roi_mean = (img[i_height - pad_height_bef:i_height+pad_height_aft+1, i_width - pad_width_bef:i_width + pad_width_aft+1].sum()/np.square((2*pad_height_bef)+1))
                roi = img[i_height - pad_height_bef:i_height+pad_height_aft+1, i_width - pad_width_bef:i_width + pad_width_aft+1]
                roi = roi - roi_mean
                corr_sum += np.sum(np.multiply(roi, template, dtype=float))
                w_sq += np.sum(np.square(roi, dtype=float))
            response[out_height][out_width] = float(corr_sum/(np.sqrt(norm_kernel, dtype=float)*np.sqrt(w_sq, dtype=float)))
    return response


###############################################
"""Helper functions: You should not have to touch the following functions.
"""
def read_img(filename):
    '''
    Read HxWxC image from the given filename
    :return img: numpy.ndarray, size (H, W, C) for RGB. The value is between [0, 255].
    '''
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_imgs(imgs, titles=None):
    '''
    Display a list of images in the notebook cell.
    :param imgs: a list of images or a single image
    '''
    if isinstance(imgs, list) and len(imgs) != 1:
        n = len(imgs)
        fig, axs = plt.subplots(1, n, figsize=(15,15))
        for i in range(n):
            axs[i].imshow(imgs[i], cmap='gray' if len(imgs[i].shape) == 2 else None)
            if titles is not None:
                axs[i].set_title(titles[i])
    else:
        img = imgs[0] if (isinstance(imgs, list) and len(imgs) == 1) else imgs
        plt.figure()
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)

def show_img_with_squares(response, img_ori=None, rec_shape=None):
    '''
    Draw small red rectangles of size defined by rec_shape around the non-zero points in the image.
    Display the rectangles and the image with rectangles in the notebook cell.
    :param response: numpy.ndarray. The input response should be a very sparse image with most of points as 0.
                     The response map is from the non-maximum suppression.
    :param img_ori: numpy.ndarray. The original image where response is computed from
    :param rec_shape: a tuple of 2 ints. The size of the red rectangles.
    '''
    response = response.copy()
    if img_ori is not None:
        img_ori = img_ori.copy()
    H, W = response.shape[:2]
    if rec_shape is None:
        h_rec, w_rec = 25, 25
    else:
        h_rec, w_rec = rec_shape

    xs, ys = response.nonzero()
    for x, y in zip(xs, ys):
        response = cv2.rectangle(response, (y - h_rec//2, x - w_rec//2), (y + h_rec//2, x + w_rec//2), (255, 0, 0), 2)
        if img_ori is not None:
            img_ori = cv2.rectangle(img_ori, (y - h_rec//2, x - w_rec//2), (y + h_rec//2, x + w_rec//2), (0, 255, 0), 2)
        
    if img_ori is not None:
        show_imgs([response, img_ori])
    else:
        show_imgs(response)

