"""Toolset for lane detection from images and camera feeds."""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from skimage import exposure

from utilityfun import plot_image, overlay_image, darken_bg, plot_text

def calibration_corner_detection(image_files, visualise=False):
    """Detect chessboard corners in input images for ensuing camera calibration."""

    # Prepare the points array with the number of points in the checkerboard pattern
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    object_points = []  # 3d points in real world space
    image_points = []   # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for image_file in image_files:
        img = cv2.imread(image_file)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret is True:
            object_points.append(objp)
            image_points.append(corners)

            # Draw and display the corners
            if visualise:
                img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

    cv2.destroyAllWindows()

    return object_points, image_points

def perspective_transform(img, src_points, reverse=False):
    """Warp the input image with a perspective transformation by taking the
    input source points and pulling them to the image corners.
    """
    img_res = (img.shape[1], img.shape[0])

    dst_top_left = [0 * img_res[0], 0 * img_res[1]]
    dst_top_right = [1 * img_res[0], 0 * img_res[1]]
    dst_bot_left = [0 * img_res[0], 1 * img_res[1]]
    dst_bot_right = [1 * img_res[0], 1 * img_res[1]]

    dst = np.float32([dst_top_left, dst_top_right, dst_bot_left,
                      dst_bot_right])

    if reverse:
        perspective_matrix = cv2.getPerspectiveTransform(dst, src_points)
    else:
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst)

    return cv2.warpPerspective(img, perspective_matrix, img_res)


def plot_color_channels(image, colorspace="RGB"):
    """Plots the colour channels of an input image."""

    print('Note: Input image must be in RGB color space!')

    if colorspace != 'RGB':
        converter = getattr(cv2, "COLOR_RGB2" + colorspace)
        converted_img = cv2.cvtColor(image, converter)
    else:
        converted_img = image
        
    color_channel_names = {'RGB': ['Red', 'Green', 'Blue']
                          ,'BGR': ['Blue', 'Green', 'Red']
                          ,'HLS': ['Hue', 'Lightness', 'Saturation']
                          ,'HSV': ['Hue', 'Saturation', 'Value']
                          ,'YUV': ['Luminance', 'Chroma U', 'Chroma V']}
    
    assert colorspace in color_channel_names, 'The chosen colorspace does not appear in our list [RGB, BGR, HLS, HSV].'

    plt.subplots(2, 2, figsize=(20,12))
    
    for i in range(1,5):
        plt.subplot(2, 2, i)
        if i==1:
            plot_image(image, title='Original Image')
        else:
            title = color_channel_names[colorspace][i-2] + ' Channel'
            plot_image(converted_img[:,:,i-2], title=title)


def translate_horz(img, direction='left', shift=0.01):
    """Translates the input image horizontally."""
    img_res = img.shape

    src_top_left = [0 * img_res[1], 0 * img_res[0]]
    src_top_right = [1 * img_res[1], 0 * img_res[0]]
    src_bot_left = [0 * img_res[1], 1 * img_res[0]]
    src_bot_right = [1 * img_res[1], 1 * img_res[0]]

    if direction is 'right':
        shift = -shift

    dst_top_left = [(0 + shift) * img_res[1], 0 * img_res[0]]
    dst_top_right = [(1 + shift) * img_res[1], 0 * img_res[0]]
    dst_bot_left = [(0 + shift) * img_res[1], 1 * img_res[0]]
    dst_bot_right = [(1 + shift) * img_res[1], 1 * img_res[0]]

    src = np.float32([src_top_left, src_top_right, src_bot_left,
                      src_bot_right])

    dst = np.float32([dst_top_left, dst_top_right, dst_bot_left,
                      dst_bot_right])

    transform = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(img, transform, (img_res[1], img_res[0]))


def abs_sobel(img, thresh_min=0, orient='x'):
    """Applies a sobel filter in x to the supplied image and discards all
    values below the threshold.
    """
    if orient == 'x':
        sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))

    scaled_sobel = 255 * sobel / np.max(sobel)
    img_output = scaled_sobel
    
    img_output = np.where(img_output > thresh_min, img_output * 5, 0)
    img_output = np.where(img_output > 255, 255, img_output)

    return normalise_image(img_output)


def combine_images(img1, img2, method="Add"):
    """Utility method to combine grayscale images."""

    possible_methods = ['Add', 'Subtract', 'Multiply']

    assert method.lower() in [x.lower() for x in possible_methods], \
        'The supplied combination method is not recognised.'

    img1 = np.array(img1, dtype=np.float32)
    img2 = np.array(img2, dtype=np.float32)

    if method.lower() == 'add':
        return normalise_image(img1 + img2)
    elif method.lower() == 'subtract':
        img2 = np.where(img2 > img1, img1, img2)
        return img1 - img2
    elif method.lower() == 'multiply':
        img_out = img1 * img2/128
        img_out = np.where(img_out > 255, 255, img_out)
        return normalise_image(img_out)
    else:
        print('Error: the supplied method is not recognised.')


def normalise_image(image, minimum=0, maximum=255):
    """Scales all values in the image so that they fit in the supplied range.
    Also ensures that output is in uint8.
    """
    norm_img = np.zeros(image.shape)
    norm_img = cv2.normalize(image, norm_img, alpha=minimum, beta=maximum, 
                             norm_type=cv2.NORM_MINMAX)
    return norm_img.astype(np.dtype('uint8'))

def dilate_and_threshold(image, threshold=50, radius=30):
    """Grow bright areas by the given radius in a grayscale image based on 
    an input threshold.
    """
    kernel = np.ones((radius, radius), np.uint8)
    thresh_image = np.where(image > threshold, 1, 0).astype('uint8')
    return cv2.dilate(thresh_image, kernel, iterations=1)


def accentuate_lane_lines(image):
    """Performs a sequence of thresholds, gradient detection and colour channel
    extractions which are then combined to intensify the lane lines.
    """
    saturation = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:, :, 2]
    red = image [:, :, 0]
    darkness = np.where((255-red) > 200, 255, 0)
    red = exposure.adjust_gamma(red, gamma=2)

    sobel_sat = abs_sobel(saturation, thresh_min=10)
    sobel_red = abs_sobel(red, thresh_min=10)
    sobel = sobel_red + sobel_sat
    sobel_shifted = (translate_horz(sobel, direction='left', shift=0.01) * 0.5 +
                     translate_horz(sobel, direction='right', shift=0.01)  * 0.5)

    combo_img = combine_images(red, sobel_shifted, method='multiply')
    combo_img = exposure.adjust_gamma(combo_img, gamma=2)

    _, norm_thresh = cv2.threshold(normalise_image(combo_img), 
                                   0, 255, 
                                   cv2.THRESH_TOZERO)
    return norm_thresh

def visualise_bins(image, vertical_bins):
    """Show the vertical bin distribution on an example image."""
    position = 0
    line_color = (0, 255, 180)

    for vertical_bin in vertical_bins:
        position += vertical_bin
        line_overlay = np.zeros(image.shape)
        line_overlay = cv2.line(line_overlay,
                                (0, position),
                                (1280, position),
                                line_color, 3)
        image = overlay_image(image, line_overlay, opacity=0.3)
    plot_image(image, title='Image Slices')


def detect_peak(img_hist, peak_pos_y, img_average, offset=0):
    
    peak_pos = np.argmax(img_hist)
    peak_val = max(img_hist)
    hist_avg = img_average
    
    confidence = 'high'
    
    if peak_val/hist_avg < 500.0:
        confidence = 'none'
    elif peak_val/hist_avg < 1500.0:
        confidence = 'low'
    elif peak_val/hist_avg < 2000.0:
        confidence = 'medium'
        
    peak_pos_x = peak_pos + offset
    
    return [peak_pos_x, peak_pos_y, confidence]


def detect_lines(img, v_bins, poly_left=None, poly_right=None):
    assert len(img.shape) == 2, 'Input image must be grayscale!'
    
    left_max =  int(img.shape[1] * 1/2)
    right_min = int(img.shape[1] * 1/2)
    
    img_mean = img.mean(axis=(0,1))
    search_margin = 15
    
    pos = 0
    left_peaks = []
    right_peaks = []
    
    for v_bin in v_bins:
        img_slice = img[pos:(pos + v_bin),:]
        histogram = np.sum(img_slice, axis=0)
        v_center = pos + int(v_bin/2)
        
        if poly_left is None:
            left_img_slice = img[pos:(pos + v_bin),:left_max]
            left_offset = 0
        else:
            l_center = poly_left[0] * v_center ** 2 + poly_left[1] * v_center ** 1 + poly_left[2]
            l_start = min(max(int(l_center) - int(search_margin),0), img.shape[1])
            l_end = min(max(int(l_center) + int(search_margin),0), img.shape[1])
            left_img_slice = img[pos:(pos + v_bin),l_start:l_end]
            left_offset = l_start
            
            #keep region of interest on image:
            l_start = max(l_start, 0)
            l_end = max(l_end, img.shape[1] + search_margin)
            l_start = min(l_start, img.shape[1] - search_margin)
            l_end = min(l_end, img.shape[1])
        
        if poly_right is None:
            right_img_slice = img[pos:(pos + v_bin),right_min:]
            right_offset = right_min
        else:
            r_center = poly_right[0] * v_center ** 2 + poly_right[1] * v_center ** 1 + poly_right[2]
            r_start = min(max(int(r_center) - int(search_margin),0), img.shape[1])
            r_end = min(max(int(r_center) + int(search_margin),0), img.shape[1])
            
            #keep region of interest on image:
            r_start = min(r_start, img.shape[1] - search_margin)
            r_end = min(r_end, img.shape[1])
            r_start = max(r_start, 0)
            r_end = max(r_end, img.shape[1] + search_margin)
            
            right_img_slice = img[pos:(pos + v_bin),r_start:r_end]
            right_offset = r_start
        
        left_hist = np.sum(left_img_slice, axis=0)
        right_hist = np.sum(right_img_slice, axis=0)

        left_peaks.append(detect_peak(left_hist, v_center, img_mean, offset=left_offset))
        right_peaks.append(detect_peak(right_hist, v_center, img_mean, offset=right_offset))
        pos += v_bin
    
    return left_peaks, right_peaks

def get_lane_poly(lane_points):
    accepted_confidence = ['high', 'medium']

    x_points, y_points, = zip(*[(float(point[0]), float(point[1])) 
                               for point in lane_points 
                               if point[2] in accepted_confidence])

    if len(x_points) < 3:
        print('hello')
        x_points = [500] * 10
        y_points = range(0, 500, 50)
    
    poly_fit = np.polyfit(y_points, x_points, 2)
    used_points = len(y_points)
    
    return poly_fit, used_points


def get_highlighted_lane(l_poly, r_poly, img, padding=400):
    y = np.linspace(0, img.shape[0]-1, img.shape[0])
    
    x_l = l_poly[0] * y ** 2 + l_poly[1] * y + l_poly[2]
    x_r = r_poly[0] * y ** 2 + r_poly[1] * y + r_poly[2]
    
    lane_color = 'lightgreen'
    
    img_overlay_lines = np.zeros((img.shape[0], img.shape[1], 3))
    img_overlay_fill = np.zeros((img.shape[0], img.shape[1], 3))
    color = (0,255, 50)
    
    # Create a polygon that includes both curves
    pts_list = list(reversed([[int(x), int(y[i])]
                  for i, x in enumerate(x_r)]))
    pts_list.extend([[int(x), int(y[i])]
                  for i, x in enumerate(x_l)])
    pts = np.array(pts_list, np.int32)
    
    img_overlay_lines = cv2.polylines(img_overlay_lines, [pts], False, color, 2)
    img_overlay_fill = cv2.fillPoly(img_overlay_fill, [pts], color)
    
    return img_overlay_lines, img_overlay_fill

def get_lane_curvature(poly, y_car):
    
    radius = (1 + (2*poly[0]*y_car + poly[1])**2)**(3/2)/abs(2*poly[0])
    
    direction = 'right' if poly[0] > 0 else 'left'
    
    return radius, direction


def get_car_offset(poly, lane, img_height, img_width, pixels_between_lanes, meters_per_pixel_x):
    x_near_car = poly[0] * img_height ** 2 + poly[1] * img_height + poly[2]
    
    if lane is 'right':
        offset = x_near_car - pixels_between_lanes/2 - img_width/2
    else:
        offset = x_near_car + pixels_between_lanes/2 - img_width/2

    direction = 'right' if offset < 0 else 'left'
    offset = abs(offset) * meters_per_pixel_x

    return offset, direction



def plot_car_offset_lines(img, offset, mpp_x, side):
    
    img_height = img.shape[0]
    img_width = img.shape[1]
    
    line_height = 30 #pixels
    
    overlay = np.zeros(img.shape, np.uint8)
    car_center_color = (0,80,255)
    lane_center_color = (255,255,255)
    
    lane_center = img_width/2 + offset/mpp_x if side == 'left' else img_width/2 - offset/mpp_x
    
    #draw car center
    cv2.line(overlay, (int(img_width / 2), img_height - line_height),
             (int(img_width / 2), img_height), car_center_color, 1)
    cv2.line(overlay, (int(lane_center), img_height - line_height),
             (int(lane_center), img_height), lane_center_color, 1)
    
    return overlay

def undistort_and_overlay(img, src_points, overlays):

    img_out = np.copy(img)

    for overlay in overlays:
        udist_overlay = perspective_transform(overlay['image'],
                                              src_points,
                                              reverse=True)
        udist_overlay = np.where(udist_overlay < 50, 0, udist_overlay)
        img_out = overlay_image(img_out, udist_overlay, 
                                opacity=overlay['opacity'])

    return img_out


def get_lane_info(img, camera_setup, src_points, vertical_sampling, timing,
                  poly_left=None, poly_right=None):
    """Lane detection pipeline function."""
    #Correct for camera distortion:
    #############################
    start = time.time()
    #############################
    udist = cv2.undistort(img, 
                          camera_setup['camera_matrix'], 
                          camera_setup['distortion_coeff'], 
                          None, 
                          camera_setup['camera_matrix'])
    #############################
    timing['distortion_correction'] += time.time() - start
    #############################
    
    #############################
    start = time.time()
    #############################

    """Tune source points to ensure that lane lines run vertically for a straight piece of road. """
    
    warped = perspective_transform(udist, src_points)
    
    #############################
    timing['perspective_transformations'] += time.time() - start
    #############################
    
    #############################
    start = time.time()
    #############################
    
    aoi = ((int(0.44 * 1280), int(0.0 * 720)),
           (int(0.57 * 1280), int(1.0 * 720))) #area of interest
    
    accentuated = np.zeros((warped.shape[0], warped.shape[1]))
    accentuated[aoi[0][1]:aoi[1][1], aoi[0][0]:aoi[1][0]] = \
        accentuate_lane_lines(warped[aoi[0][1]:aoi[1][1], aoi[0][0]:aoi[1][0], :])
    
    #############################
    timing['image_manipulation'] += time.time() - start
    #############################
    
    #############################
    start = time.time()
    #############################
    #Detect lane markings
    left_peaks, right_peaks = detect_lines(accentuated, vertical_sampling, 
                                           poly_left=poly_left, poly_right=poly_right)
    
    #Fit polynomials to result
    left_lane_poly, points_left = get_lane_poly(left_peaks)
    right_lane_poly, points_right = get_lane_poly(right_peaks)

    if points_left < 6:
        print("Left")
        left_lane_poly = poly_left
    if points_right < 6:
        print("Right")
        right_lane_poly = poly_right
    
    
    #Get lane corner radius and car offset
    mpp_x = 3.7/camera_setup['pixels_between_lanes'] #meters per pixel in x-direction
    mpp_y = 3.0/camera_setup['pixels_along_dash'] #meters per pixel in y-direction
    
    coeff_scaling = [mpp_x / (mpp_y ** 2), (mpp_x/mpp_x), 1]

    dominant_poly, lane = (left_lane_poly, 'left') if \
        points_left > points_right else (right_lane_poly, 'right')
    poly_real = [coeff * coeff_scaling[i] for i, coeff in enumerate(dominant_poly)]

    lane_radius, direction = get_lane_curvature(poly_real, warped.shape[0]*mpp_y)
    lane_rad_string = "Radius: " + "{:1.2f}".format(lane_radius) + 'm to the ' + direction 
    
    car_offset, side = get_car_offset(dominant_poly, lane, warped.shape[0], warped.shape[1], 
                                      camera_setup['pixels_between_lanes'], mpp_x)
    car_pos_string = "Offset: " + "{:1.2f}".format(car_offset) + 'm to the ' + side 
    
    #############################
    timing['polynomial_fitting'] += time.time() - start
    #############################
    
    #############################
    start = time.time()
    #############################
    
    #Create image overlay for output
    overlay_lines, overlay_fill = get_highlighted_lane(left_lane_poly, right_lane_poly, accentuated)
    overlay_offset = plot_car_offset_lines(warped, car_offset, mpp_x, side)
    
    # undistort lane detection annotations
    overlays = ({'image': overlay_lines,  'opacity': 0.8},
                {'image': overlay_fill,   'opacity': 0.3},
                {'image': overlay_offset, 'opacity': 0.8},)

    img_with_overlay = undistort_and_overlay(udist, src_points, overlays)
    
    #############################
    timing['perspective_transformations'] += time.time() - start
    #############################
    
    #############################
    start = time.time()
    #############################

    img_with_overlay = darken_bg(img_with_overlay, img.shape[0] - 40,  img.shape[0], 0, img.shape[1])
    img_with_overlay = plot_text(img_with_overlay, lane_rad_string, 'left') 
    img_with_overlay = plot_text(img_with_overlay, car_pos_string, 'right')
    #############################
    timing['annotating_images'] += time.time() - start
    #############################
    
    return img_with_overlay, left_lane_poly, right_lane_poly, timing