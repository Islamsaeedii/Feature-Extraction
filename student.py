
import numpy as np   
import cv2 
from scipy.signal import convolve2d




def get_interest_points(image, feature_width=None):
    """
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation 
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!
    bluring=cv2.GaussianBlur(image, ksize=(13, 1), sigmaX=3, sigmaY=0.5) # to remove noise  
    gradiaant_xx, gradiaant_yy = np.gradient(bluring) # to get the gradient of the image
    y_square = gradiaant_yy**2
    y_z_mult = gradiaant_yy*gradiaant_xx
    gradiaant_xx_2 = gradiaant_xx**2
    grad_x=cv2.GaussianBlur(y_square, ksize=(13, 1), sigmaX=3, sigmaY=0.5)
    dltal_xy=cv2.GaussianBlur(y_z_mult, ksize=(13, 1), sigmaX=3, sigmaY=0.5)
    grad_y=cv2.GaussianBlur(gradiaant_xx_2, ksize=(13, 1), sigmaX=3, sigmaY=0.5)

    det_A = grad_x * grad_y - dltal_xy **2
    trace_A = grad_x + grad_y
    harriiis = det_A -0.05 * trace_A ** 2

    px1, py1 = harriiis.shape
    final_points = []
    for feature in range(0, px1):
        indxxx = []
        for feature2 in range(0, py1):
                indxxx = [harriiis[feature,feature2]]
                indxxx.append(feature2)
                indxxx.append(feature)
                final_points.append(indxxx)

    final_sorted = sorted(final_points, reverse = True)
    first_n_best_interst_point = 2000
    final_sorted = final_sorted[0:first_n_best_interst_point]
    get_coordinates = []

    for i in range(0, first_n_best_interst_point):
        get_coordinates.append([float('inf'), final_sorted[i][0], final_sorted[i][1], final_sorted[i][2]])
    x_corrdinates = []
    y_corrdinates = []
    for interest_point_neighborhood in sorted(get_coordinates, reverse = True):
        x_corrdinates.append(interest_point_neighborhood[2])
        y_corrdinates.append(interest_point_neighborhood[3])  
    x = np.array(x_corrdinates)
    y = np.array(y_corrdinates)

    return x,y

def get_features(image, x, y, feature_width):
    """
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized windows10es as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a hist of the local distribution of
        gradients in 8 angles. Appending these hists together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular angles you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    """
    # TODO: Your implementation here! See block comments and the project webpage for instructions

    num_points = x.shape[0]
    features = np.zeros((num_points, 128))
   
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    blured = cv2.GaussianBlur(image, ksize=(11, 1), sigmaX=1, sigmaY=0.5)
    Ix = convolve2d(blured, sobel_x, 'same') 
    Iy = convolve2d(blured, sobel_y, 'same')  
    angles = np.arctan2(Iy, Ix)
    gradiantss = np.sqrt((Ix ** 2) + (Iy ** 2))

    angles[angles < 0] += 2 * np.pi
    L = int(feature_width / 2)
    ind=[i for i in range(len(x))]
    padding1 = int(feature_width/2)
    padding2 = int((feature_width/2)+1)
    blured=np.pad(blured, ((padding1,padding2), (padding1,padding2)), 'edge')
    tem=x
    x=y
    y=tem
    for x_point, y_point ,dd in zip(x, y,ind):
        x_point=x_point.astype(int)
        y_point=y_point.astype(int)
        windows10 = blured[x_point-L:x_point+L, y_point-L:y_point+L]
        if y_point>blured.shape[0] or x_point>blured.shape[1] or y_point <8 or x_point<8:
            continue
        gradian_ = gradiantss[x_point-L:x_point+L, y_point-L:y_point+L]
        rotation = angles[x_point-L:x_point+L, y_point-L:y_point+L]
        
        hist = []
        for iy in range(0, windows10.shape[0], 4):
            for ix in range(0, windows10.shape[1], 4):
                grad_small = np.array(gradian_[iy:iy + 4, ix:ix + 4]).flatten()
                orian_small = np.array(rotation[iy:iy + 4, ix:ix + 4]).flatten()
                histo = np.zeros((1, 8))
                for i in range(orian_small.shape[0]):
                    index = min(int(orian_small[i]/(2 * np.pi/8)), 7)
                    histo[0, index] += grad_small[i]

                hist.extend(histo[0])

        hist = np.array(hist).reshape(1, -1)
        hist_norm = np.linalg.norm(hist)
        hist /= hist_norm
        hist=np.where(hist>0.056,0.056,hist)
        features[dd, :] = hist


    return features



def match_features(im1_features, im2_features):
    """
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar desctinses.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with your matches and confidences!
    #print(im1_features.shape,im2_features.shape)
    #pca = PCA(n_components=120)
    #im1_features=pca.fit_transform(im1_features)
    #im2_features=pca.fit_transform(im2_features)

    matches = np.zeros((1, 2))
    confidences = np.zeros(1)

    matches = []
    confidences = []
    thresh = 0.95
    desctinses=[]
    index_s=[a for a in range(im1_features.shape[0])]
    for i,feature1 in zip(index_s,im1_features):
        desctinses=[]
        for feature2 in im2_features:
            desctinses.append(np.linalg.norm(feature1-feature2))
        indexes=sorted(range(len(desctinses)), key=desctinses.__getitem__)
        d1 = desctinses[indexes[0]]
        d2 = desctinses[indexes[1]]
        NNDR=d1/d2
        if NNDR < thresh:
            matches.append([i, indexes[0]])
            confidences.append(1- NNDR)
    matches = np.array(matches)
    confidences = np.array(confidences)
    return matches, confidences


