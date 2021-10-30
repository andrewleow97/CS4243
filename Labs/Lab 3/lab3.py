import numpy as np
from numpy.core.fromnumeric import diagonal
from skimage import filters
from skimage.feature import corner_peaks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter
import math

### REMOVE THIS
from cv2 import IMWRITE_EXR_TYPE, detail_AffineBasedEstimator, findHomography
from skimage.filters.edges import sobel_h, sobel_v

from utils import pad, unpad

import cv2
_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

##################### PART 1 ###################

# 1.1 IMPLEMENT
def harris_corners(img, window_size=3, k=0.04):
    '''
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the functions filters.sobel_v filters.sobel_h & scipy.ndimage.filters.convolve, 
        which are already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    '''

    H, W= img.shape
    window = np.ones((window_size, window_size))
    response = np.zeros((H, W))

    # YOUR CODE HERE

    Ix = filters.sobel_h(img) 
    Iy = filters.sobel_v(img) 

    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    Ixy = np.multiply(Ix, Iy)
    # To get A, B and C per window, convolve Sobel filter response of image with window 
    A = convolve(Ix2, window, mode='constant', cval=0)
    B = convolve(Ixy, window, mode='constant', cval=0)
    C = convolve(Iy2, window, mode='constant', cval=0)

    det = np.subtract(np.multiply(A, C), np.square(B))
    trace = np.add(A, C)
    response = det - k * np.square(trace)
    # END        
    return response

# 1.2 IMPLEMENT
def naive_descriptor(patch):
    '''
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.

    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    '''
    feature = []
    ### YOUR CODE HERE
    mean = np.mean(patch)
    stdev = np.std(patch)
    F = (patch - mean)/(stdev + 0.0001)
    feature = F.flatten() # feature: 1D array of shape (h * w)
    ### END YOUR CODE

    return feature

# GIVEN
def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    '''
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (x, y) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    '''

    image.astype(np.float32)
    desc = []
    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[np.max([0,y-(patch_size//2)]):y+((patch_size+1)//2),
                      np.max([0,x-(patch_size//2)]):x+((patch_size+1)//2)]
      
        desc.append(desc_func(patch))
   
    return np.array(desc)

# GIVEN
def make_gaussian_kernel(ksize, sigma):
    '''
    Good old Gaussian kernel.
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    '''

    ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    yy, xx = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(yy) + np.square(xx)) / np.square(sigma))

    return kernel / kernel.sum()


# 1.2 IMPLEMENT
def simple_sift(patch):
    '''
    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each length of 16/4=4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    Use the gradient orientation to determine the bin, and the gradient magnitude * weight from
    the Gaussian kernel as vote weight.

    Args:
        patch: grayscale image patch of shape (h, w)

    Returns:
        feature: 1D array of shape (128, )
    '''
    
    # You can change the parameter sigma, which has been default to 3
    weights = np.flipud(np.fliplr(make_gaussian_kernel(patch.shape[0],3)))
    
    histogram = np.zeros((4,4,8))
    
    H,W = patch.shape

    # YOUR CODE HERE
    Ix = sobel_h(patch)
    Iy = sobel_v(patch)
    gradient_magnitude = np.hypot(Ix, Iy)
    gradient_orientation = np.arctan2(Iy, Ix) * 180/math.pi + 180
    for i in range(0, H, 4):
        for j in range(0, W, 4): # 4x4 cells
            for row in range(4):
                for col in range(4): # pixels in cell
                    angle = gradient_orientation[i+row][j+col]
                    if 0 <= angle < 45: # getting bin index
                        bin = 0
                    elif 45 <= angle < 90:
                        bin = 1
                    elif 90 <= angle < 135:
                        bin = 2
                    elif 135 <= angle < 180:
                        bin = 3
                    elif 180 <= angle < 225:
                        bin = 4
                    elif 225 <= angle < 270:
                        bin = 5
                    elif 270 <= angle < 315:
                        bin = 6
                    elif 315 <= angle < 360:
                        bin = 7
                    else: # angle = 360
                        bin = 0          
                    weight = gradient_magnitude[i+row][j+col] * weights[i+row][j+col]
                    histogram[int(i/4), int(j/4), bin] += weight
    feature = histogram.flatten()
    feature = feature / np.linalg.norm(feature)
    # END
    return feature

# 1.3 IMPLEMENT
def top_k_matches(desc1, desc2, k=2):
    '''
    Compute the Euclidean distance between each descriptor in desc1 versus all descriptors in desc2 (Hint: use cdist).
    For each descriptor Di in desc1, pick out k nearest descriptors from desc2, as well as the distances themselves.
    Example of an output of this function:
    
        [(0, [(18, 0.11414082134194799), (28, 0.139670625444803)]),
         (1, [(2, 0.14780585099287238), (9, 0.15420019834435536)]),
         (2, [(64, 0.12429203239414029), (267, 0.1395765079352806)]),
         ...<truncated>
    '''
    match_pairs = []
    # YOUR CODE HERE
    euc_dist = cdist(desc1, desc2, metric='euclidean')

    for i in range(euc_dist.shape[0]): # iterate through each desc1 euc_dist
        dist = np.argsort(euc_dist[i])[:k] # k indices of smallest distances in euc_dist
        nearest = []
        for j in range(len(dist)):
            nearest.append((dist[j], euc_dist[i][dist[j]])) # append (index, value @ index)
        match_pairs.append((i, np.array(nearest)))
    
    # END
    return match_pairs

# 1.3 IMPLEMENT
def ratio_test_match(desc1, desc2, match_threshold):
    '''
    Match two set of descriptors using the ratio test.
    Output should be a numpy array of shape (k,2), where k is the number of matches found. 
    In the following sample output:
        array([[  3,   0],
               [  5,  30],
               [ 11,   9],
               [ 18,   7],
               [ 24,   5],
               [ 30,  17],
               [ 32,  24],
               [ 46,  23], ... <truncated>
              )
              
        desc1[3] is matched with desc2[0], desc1[5] is matched with desc2[30], and so on.
    
    All other match functions will return in the same format as does this one.
    
    '''
    match_pairs = []
    top_2_matches = top_k_matches(desc1, desc2)
    # YOUR CODE HERE
    for i in top_2_matches: # i of the form tuple (desc1_index, [(desc2_index0, value0), (desc2_index1, value1)])
        F1i = i[0]
        F2 = i[1]
        F2a = F2[0]
        F2b = F2[1]
        ratio = np.divide(F2a[1], F2b[1])
        if ratio < match_threshold:
            match_pairs.append([F1i, F2a[0]])
    # END
    # Modify this line as you wish
    match_pairs = np.array(match_pairs, dtype=int)
    return match_pairs

# GIVEN
def compute_cv2_descriptor(im, method=cv2.SIFT_create()):
    '''
    Detects and computes keypoints using one of the implementations in OpenCV
    You can use:
        cv2.SIFT_create()

    Do note that the keypoints coordinate is (col, row)-(x,y) in OpenCV. We have changed it to (row,col)-(y,x) for you. (Consistent with out coordinate choice)
    '''
    kpts, descs = method.detectAndCompute(im, None)
    
    keypoints = np.array([(kp.pt[1],kp.pt[0]) for kp in kpts])
    angles = np.array([kp.angle for kp in kpts])
    sizes = np.array([kp.size for kp in kpts])
    
    return keypoints, descs, angles, sizes

##################### PART 2 ###################

# GIVEN
def transform_homography(src, h_matrix, getNormalized = True):
    '''
    Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    '''
    transformed = None

    input_pts = np.insert(src, 2, values=1, axis=1)
    transformed = np.zeros_like(input_pts)
    transformed = h_matrix.dot(input_pts.transpose())
    if getNormalized:
        transformed = transformed[:-1]/transformed[-1]
    transformed = transformed.transpose().astype(np.float32)
    
    return transformed

# 2.1 IMPLEMENT
def compute_homography(src, dst):
    '''
    Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    '''
    h_matrix = np.eye(3, dtype=np.float64)


    # YOUR CODE HERE

    # Normalize X = src by setting to (0,0) and distance to origin = sqrt(2)
    # Normalize X' = dst as well, same steps
    # DLT to get H'
    # Denormalization H = T'^-1 H' T

    N = src.shape[0]
    src = pad(src)
    dst = pad(dst)
    # T = [[1/sx, 0, -mx/sx], [0, 1/sy, -my/sy], [0, 0, 1]]

    src_mx, src_my, _ = np.mean(src, axis=0) 
    src_stdx, src_stdy, _ = np.std(src, axis=0)/np.sqrt(2)
    T_src = [[1/src_stdx, 0, -1 * src_mx/src_stdx], 
            [0, 1/src_stdy, -1 * src_my/src_stdy],
            [0, 0, 1]]
    src_norm = np.matmul(T_src, src.T).T # 3 x N matrix

    dst_mx, dst_my, _ = np.mean(dst, axis=0) 
    dst_stdx, dst_stdy, _ = np.std(dst, axis=0)/np.sqrt(2)
    T_dst = [[1/dst_stdx, 0, -1 * dst_mx/dst_stdx], 
            [0, 1/dst_stdy, -1 * dst_my/dst_stdy],
            [0, 0, 1]]
    dst_norm = np.matmul(T_dst, dst.T).T # 3 x N matrix

    # calculating homography matrix A
    A = []
    for i in range(N):
        x = src_norm[i][0]
        y = src_norm[i][1]
        xp = dst_norm[i][0]
        yp = dst_norm[i][1]
        Ai = np.array([[-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp],
                        [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]])
        A.append(Ai)
    A = np.concatenate(A, axis=0) # 2N x 9 matrix

    # Getting SVD of A
    u, s, vh = np.linalg.svd(A)

    # Store singular vector of smallest singular value
    k = vh[-1].reshape((3, 3)) # 3x3 matrix

    # Get H using de-normalization
    kT = np.dot(k, T_src) # 3xN matrix
    h_matrix = np.dot(np.linalg.inv(T_dst), kT)
    # END 
    return h_matrix

# 2.2 IMPLEMENT
def ransac_homography(keypoints1, keypoints2, matches, sampling_ratio=0.5, n_iters=500, delta=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        sampling_ratio: percentage of points selected at each iteration
        n_iters: the number of iterations RANSAC will run
        threshold: the threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * sampling_ratio)

    matched1_unpad = keypoints1[matches[:,0]]
    matched2_unpad = keypoints2[matches[:,1]]


    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    ### YOUR CODE HERE
    for i in range(n_iters):
        random_sample = np.random.choice(N, n_samples)
        src = matched1_unpad[random_sample] # leave unpadded, is padded in compute homography
        dst = matched2_unpad[random_sample]

        # Get H using DLT
        H_matrix = compute_homography(src, dst)

        # Matched keypoints are considered inliers if the projected point from one image lies within distance δ to the matched point on the other point.

        projection = transform_homography(matched1_unpad, H_matrix) # x point, y point
        distances = np.sqrt(np.sum(np.square(projection-matched2_unpad), axis=1))
        if np.sum(distances < delta) > n_inliers:
            max_inliers = distances < delta
            n_inliers = np.sum(distances < delta)
            
    # Get final H using maximum inliers indices and src dst values
    src = matched1_unpad[max_inliers]
    dst = matched2_unpad[max_inliers]
    H = compute_homography(src, dst)
    
    ### END YOUR CODE
    return H, matches[max_inliers]



##################### PART 3 ###################
# GIVEN FROM PREV LAB
from skimage.feature import peak_local_max
def find_peak_params(hspace, params_list,  window_size=1, threshold=0.5):
    '''
    Given a Hough space and a list of parameters range, compute the local peaks
    aka bins whose count is larger max_bin * threshold. The local peaks are computed
    over a space of size (2*window_size+1)^(number of parameters)

    Also include the array of values corresponding to the bins, in descending order.
    '''
    assert len(hspace.shape) == len(params_list), \
        "The Hough space dimension does not match the number of parameters"
    for i in range(len(params_list)):
        assert hspace.shape[i] == len(params_list[i]), \
            f"Parameter length does not match size of the corresponding dimension:{len(params_list[i])} vs {hspace.shape[i]}"
    peaks_indices = peak_local_max(hspace.copy(), exclude_border=False, threshold_rel=threshold, min_distance=window_size)
    peak_values = np.array([hspace[tuple(peaks_indices[j])] for j in range(len(peaks_indices))])
    res = []
    res.append(peak_values)
    for i in range(len(params_list)):
        res.append(params_list[i][peaks_indices.T[i]])
    return res

# GIVEN
def angle_with_x_axis(pi, pj):  
    '''
    Compute the angle that the line connecting two points I and J make with the x-axis (mind our coordinate convention)
    Do note that the line direction is from point I to point J.
    '''
    # get the difference between point p1 and p2
    y, x = pi[0]-pj[0], pi[1]-pj[1] 
    
    if x == 0:
        return np.pi/2  
    
    angle = np.arctan(y/x)
    if angle < 0:
        angle += np.pi
    return angle

# GIVEN
def midpoint(pi, pj):
    '''
    Get y and x coordinates of the midpoint of I and J
    '''
    return (pi[0]+pj[0])/2, (pi[1]+pj[1])/2

# GIVEN
def distance(pi, pj):
    '''
    Compute the Euclidean distance between two points I and J.
    '''
    y,x = pi[0]-pj[0], pi[1]-pj[1] 
    return np.sqrt(x**2+y**2)

# 3.1 IMPLEMENT
def shift_sift_descriptor(desc):
    '''
       Generate a virtual mirror descriptor for a given descriptor.
       Note that you have to shift the bins within a mini histogram, and the mini histograms themselves.
       e.g:
       Descriptor for a keypoint
       (the dimension is (128,), but here we reshape it to (16,8). Each length-8 array is a mini histogram.)
      [[  0.,   0.,   0.,   5.,  41.,   0.,   0.,   0.],
       [ 22.,   2.,   1.,  24., 167.,   0.,   0.,   1.],
       [167.,   3.,   1.,   4.,  29.,   0.,   0.,  12.],
       [ 50.,   0.,   0.,   0.,   0.,   0.,   0.,   4.],
       
       [  0.,   0.,   0.,   4.,  67.,   0.,   0.,   0.],
       [ 35.,   2.,   0.,  25., 167.,   1.,   0.,   1.],
       [167.,   4.,   0.,   4.,  32.,   0.,   0.,   5.],
       [ 65.,   0.,   0.,   0.,   0.,   0.,   0.,   1.],
       
       [  0.,   0.,   0.,   0.,  74.,   1.,   0.,   0.],
       [ 36.,   2.,   0.,   5., 167.,   7.,   0.,   4.],
       [167.,  10.,   0.,   1.,  30.,   1.,   0.,  13.],
       [ 60.,   2.,   0.,   0.,   0.,   0.,   0.,   1.],
       
       [  0.,   0.,   0.,   0.,  54.,   3.,   0.,   0.],
       [ 23.,   6.,   0.,   4., 167.,   9.,   0.,   0.],
       [167.,  40.,   0.,   2.,  30.,   1.,   0.,   0.],
       [ 51.,   8.,   0.,   0.,   0.,   0.,   0.,   0.]]
     ======================================================
       Descriptor for the same keypoint, flipped over the vertical axis
      [[  0.,   0.,   0.,   3.,  54.,   0.,   0.,   0.],
       [ 23.,   0.,   0.,   9., 167.,   4.,   0.,   6.],
       [167.,   0.,   0.,   1.,  30.,   2.,   0.,  40.],
       [ 51.,   0.,   0.,   0.,   0.,   0.,   0.,   8.],
       
       [  0.,   0.,   0.,   1.,  74.,   0.,   0.,   0.],
       [ 36.,   4.,   0.,   7., 167.,   5.,   0.,   2.],
       [167.,  13.,   0.,   1.,  30.,   1.,   0.,  10.],
       [ 60.,   1.,   0.,   0.,   0.,   0.,   0.,   2.],
       
       [  0.,   0.,   0.,   0.,  67.,   4.,   0.,   0.],
       [ 35.,   1.,   0.,   1., 167.,  25.,   0.,   2.],
       [167.,   5.,   0.,   0.,  32.,   4.,   0.,   4.],
       [ 65.,   1.,   0.,   0.,   0.,   0.,   0.,   0.],
       
       [  0.,   0.,   0.,   0.,  41.,   5.,   0.,   0.],
       [ 22.,   1.,   0.,   0., 167.,  24.,   1.,   2.],
       [167.,  12.,   0.,   0.,  29.,   4.,   1.,   3.],
       [ 50.,   4.,   0.,   0.,   0.,   0.,   0.,   0.]]
    '''
    # YOUR CODE HERE
    #transforming a 128-length array to another 128-length array.
    # SIFT already shifts the indices to keep the dominant orientation first, the first index (0) will stay in the same position. Remaining bins are reversed, i.e. [0, 1, 2, ..7] remaps to [0, 7, 6, .., 2, 1].
    
    unmirrored = desc.reshape((16,8))
    mirrored = []
    for i in range(unmirrored.shape[0]): # for each in 16
        l = unmirrored[i][0] # first term leave unflipped
        l = np.append(l,np.flip(unmirrored[i][1:8])) # flip the rest of the terms
        mirrored.append(list(l))
    mirrored = np.flipud(np.array(mirrored))
    res = []
    for i in range(0,16,4):
        a = np.flipud(mirrored[i:i+4]) # flipping in grps of 4
        for j in range(4):
            res.append(a[j])
    res = np.array(res)
    res = res.reshape(128) # make back into 128-length array

    return res
    # END

# 3.1 IMPLEMENT
def create_mirror_descriptors(img):
    '''
    Return the output for compute_cv2_descriptor (which you can find in utils.py)
    Also return the set of virtual mirror descriptors.
    Make sure the virtual descriptors correspond to the original set of descriptors.
    '''
    # YOUR CODE HERE
    kps, descs, sizes, angles = compute_cv2_descriptor(img, method=cv2.SIFT_create())
    mir_descs = []
    for i in range(descs.shape[0]):
        mir_descs.append(shift_sift_descriptor(descs[i]))
    # END
    mir_descs = np.array(mir_descs) # should be same shape as descs
    return kps, descs, sizes, angles, mir_descs

# 3.2 IMPLEMENT
def match_mirror_descriptors(descs, mirror_descs, threshold = 0.7):
    '''
    First use `top_k_matches` to find the nearest 3 matches for each keypoint. Then eliminate the mirror descriptor that comes 
    from the same keypoint. Perform ratio test on the two matches left. If no descriptor is eliminated, perform the ratio test 
    on the best 2. 
    '''
    match_result = []
    top_three = top_k_matches(descs, mirror_descs, k=3)
    top_three = np.array(top_three)
    top_two = []
    for kp, match in top_three:
        if kp == int(match[0][0]): # mirror descriptor, eliminate
            new_list = match[1:]
        else:
            new_list = match[:2] # best 2, should be first two since dist is sorted
        top_two.append((kp, new_list))

    top_two = np.array(top_two)
    for i in top_two: # i of the form tuple (desc1_index, [(desc2_index0, value0), (desc2_index1, value1)])
        F1i = i[0]
        F2 = i[1]
        F2a = F2[0]
        F2b = F2[1]

        ratio = np.divide(F2a[1], F2b[1])
        if ratio < threshold:
            match_result.append([F1i, F2a[0]])

    match_result = np.array(match_result, dtype=int)
    # END     

    return match_result

# 3.3 IMPLEMENT
def find_symmetry_lines(matches, kps):
    '''
    For each pair of matched keypoints, use the keypoint coordinates to compute a candidate symmetry line.
    Assume the points associated with the original descriptor set to be I's, and the points associated with the mirror descriptor set to be J's.
    '''
    rhos = []
    thetas = []
    # YOUR CODE HERE
    #  For a matched pair of points 𝑖 and 𝑘′, form a line that intersects the two points; this line is perpendicular to the symmetry line. Solve for the midpoint 𝑚=(𝑥𝑚,𝑦𝑚) between 𝑖 and 𝑘′ on the intersecting line and compute the angle 𝜃𝑚 that the line makes with the x-axis.
    for match in matches:
        I = match[0] # index of I
        J = match[1] # index of J
        ptI = kps[I]
        ptJ = kps[J]
        
        mid = midpoint(ptI, ptJ)
        angle = angle_with_x_axis(ptI, mid)

        rho = mid[1] * np.cos(angle) + mid[0] * np.sin(angle)
        rhos.append(rho)
        thetas.append(angle)
    #END
    return rhos, thetas

# 3.4 IMPLEMENT
def hough_vote_mirror(matches, kps, im_shape, window=1, threshold=0.5, num_lines=1, interval=1):
    '''
    Hough Voting:
                 0<=thetas<= 2pi      , interval size = 1 degree
        -diagonal <= rhos <= diagonal , interval size = 1 pixel
    Feel free to vary the interval size.
    '''
    
    rhos, thetas = find_symmetry_lines(matches, kps)
    rhos = np.array(rhos)
    thetas = np.array(thetas) # in radian
    # # YOUR CODE HERE

    # # Setting up A
    diag_interval = interval
    theta_interval = interval*math.pi/180

    diagonal = round(np.hypot(im_shape[0], im_shape[1]))
    diag_dim = np.arange(-diagonal, diagonal, diag_interval)
    theta_dim = np.arange(0, 2*math.pi, theta_interval)
    A = np.zeros((diag_dim.shape[0], theta_dim.shape[0]))

    for i in range(len(rhos)):
        rho = np.argmax(diag_dim > rhos[i])
        theta = np.argmax(theta_dim > thetas[i])
        A[rho][theta] += 1

    # The line(s) of symmetry in the image will be represented by local maxima in the Hough vote space.
    params = [diag_dim, theta_dim]
    peak = find_peak_params(A, params, window, threshold)
    rho_values = []
    theta_values = []
    for i in range(num_lines):
        rho_values.append(peak[1][i])
        theta_values.append(peak[2][i])

    
    rho_values = np.array(rho_values)
    theta_values = np.array(theta_values)
    # END
    
    return rho_values, theta_values


##################### PART 4 ###################

# 4.1 IMPLEMENT
def match_with_self(descs, kps, threshold=0.8):
    '''
    Use `top_k_matches` to match a set of descriptors against itself and find the best 3 matches for each descriptor.
    Discard the trivial match for each trio (if exists), and perform the ratio test on the two matches left (or best two if no match is removed)
    '''
   
    matches = []
    
    # YOUR CODE HERE
    top_three = top_k_matches(descs, descs, k=3)
    top_three = np.array(top_three)
    top_two = []
    for kp, match in top_three:
        new_list = []
        if kp == (match[0][0]):
            new_list = match[1:] # match with itself
        else:
            new_list = match # no match with itself
        top_two.append((kp, new_list))
    top_two = np.array(top_two)

    for i in top_two: # i of the form tuple (desc1_index, [(desc2_index0, value0), (desc2_index1, value1)])
        F1i = i[0]
        F2 = i[1]
        F2a = F2[0]
        F2b = F2[1]

        ratio = np.divide(F2a[1], F2b[1])
        if ratio < threshold:
            matches.append([F1i, F2a[0]])

    matches = np.array(matches, dtype=int)
    # END     
    return matches

# 4.2 IMPLEMENT
def find_rotation_centers(matches, kps, angles, sizes, im_shape):
    '''
    For each pair of matched keypoints (using `match_with_self`), compute the coordinates of the center of rotation and vote weight. 
    For each pair (kp1, kp2), use kp1 as point I, and kp2 as point J. The center of rotation is such that if we pivot point I about it,
    the orientation line at point I will end up coinciding with that at point J. 
    
    You may want to draw out a simple case to visualize first.
    
    If a candidate center lies out of bound, ignore it.
    '''
    # Y-coordinates, X-coordinates, and the vote weights 
    Y = []
    X = []
    W = []
    
    # YOUR CODE HERE
    angles = angles % 360 # make into range [0,360), convert to radians later
    for match in matches:
        I = match[0] # point to pivot I
        J = match[1] # centre J
        ptI = kps[I] # array of x-y coords
        ptJ = kps[J] # array of x-y coords
 
        angleI = (angles[I]  * math.pi/180) # single radian value
        angleJ = (angles[J]  * math.pi/180) # single radian value
        sizeI = sizes[I] # single size value
        sizeJ = sizes[J] # single size value

        if (abs(angleI-angleJ) <= (1 * math.pi/180)): # parallel angles within 1 degree radian, discard
            continue

        d = distance(ptI, ptJ)
        y = angle_with_x_axis(ptI, ptJ)

        B = (angleI - angleJ + math.pi)/2 
        r = (d * np.sqrt(1 + np.square(np.tan(B)))) / 2 

        Xc = round(ptI[1] + r * np.cos(B + y))
        Yc = round(ptI[0] + r * np.sin(B + y))

        if (Xc < 0 or Xc > im_shape[1] or Yc < 0 or Yc > im_shape[0]): # check if in range
            continue

        q = -1 * abs(sizeI - sizeJ) / (sizeI + sizeJ)
        weight = np.square(np.exp(q)) # w = (e^q)^2
        Y.append(Yc)
        X.append(Xc)
        W.append(weight)
        
    # END
    
    return Y,X,W

# 4.3 IMPLEMENT
def hough_vote_rotation(matches, kps, angles, sizes, im_shape, window=1, threshold=0.5, num_centers=1, interval=1):
    '''
    Hough Voting:
        X: bound by width of image
        Y: bound by height of image
    Return the y-coordianate and x-coordinate values for the centers (limit by the num_centers)
    '''
    
    Y,X,W = find_rotation_centers(matches, kps, angles, sizes, im_shape)
    y_values = []
    x_values = []
    # YOUR CODE HERE
    A = np.zeros((math.ceil(im_shape[0]/interval), math.ceil(im_shape[1]/interval)), dtype=float) # y is first term x is second term
    y_range = np.arange(0, im_shape[0], interval)
    x_range = np.arange(0, im_shape[1], interval)
    for i in range(len(X)):
        y = np.argmax(y_range > Y[i])
        x = np.argmax(x_range > X[i])
        A[y][x] += W[i] # add weight to position in accumulator array

    params = [y_range, x_range]
    peak = find_peak_params(A, params, window, threshold)

    # Solution with peak params
    for i in range(num_centers):
        y_values.append(peak[1][i])
        x_values.append(peak[2][i])

    # Solution with self-implemented global maxima

    # for i in range(num_centers):
    #     # The global maxima should correspond to the center of rotation.
    #     # Basically do non-max suppression for num_centers times
    #     max_val = np.amax(A)
    #     centers = np.where(A == max_val) # indices of max val, append to y,x
    #     y = centers[0][0]
    #     x = centers[1][0]
    #     y_values.append(y*interval)
    #     x_values.append(x*interval)
    #     A[y][x] = 0

    return y_values, x_values
