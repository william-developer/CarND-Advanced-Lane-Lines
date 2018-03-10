import numpy as np
import cv2

def abs_sobel_thresh(image, orient='x', thresh_min=0, thresh_max=255):
    '''
    Function that applies Sobel x or y,
    then takes an absolute value and applies a threshold.
    Note: calling your function with orient='x', thresh_min=5, thresh_max=100
    should produce output like the example image shown above this quiz.
    Define a function that takes an image, gradient orientation,
    and threshold min / max values.
    '''
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    '''
    Function that applies Sobel x and y,
    then computes the magnitude of the gradient
    and applies a threshold
    Define a function to return the magnitude of the gradient
    for a given sobel kernel size and threshold values
    '''
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    '''
    Function that applies Sobel x and y,
    then computes the direction of the gradient
    and applies a threshold.
    Define a function to threshold an image for a given range and Sobel kernel
    '''
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_select(img, channel='s',thresh=(0, 255)):
    '''hls color thresholds'''
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel=='s':
        s_channel = hls[:,:,2]
    elif channel=='h':
        s_channel = hls[:,:,0]
    elif channel=='l':
        s_channel = hls[:,:,1]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output
    
def combine_gradient(image,s_thresh=(170, 255), sx_thresh=(20, 100),l_thresh=(100,255)):
    '''Combined S channel and gradient thresholds'''
    # Convert to HLS color space and separate the S channel
    s_binary = hls_select(image, channel='s',thresh=s_thresh)

    # Convert to HLS color space and separate the L channel
    l_binary = hls_select(image, channel='l',thresh=l_thresh)
    
    # Sobel x
    (thresh_min,thresh_max) = sx_thresh
    sxbinary = abs_sobel_thresh(image, orient='x', thresh_min=thresh_min, thresh_max=thresh_max)
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1)&(l_binary ==1)) | (sxbinary == 1)] = 1
    return combined_binary

def warp_perspective(image,mtx,dist):
    '''perspective by taking an image, number of x and y points, camera matrix and distortion coefficients'''
    img_size = (image.shape[1], image.shape[0])
    w = img_size[0]
    h = img_size[1]
    # Given src and dst points, calculate the perspective transform matrix
    src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 80), img_size[1] / 2 + 100]])
    dst = np.float32(
    [[(img_size[0] / 7), 0],
    [(img_size[0] / 7), img_size[1]],
    [(img_size[0] * 4 / 5), img_size[1]],
    [(img_size[0] * 4 / 5), 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(image, M, img_size,flags=cv2.INTER_LINEAR)
    return warped, M

def polynomial(image):
    '''Fit left and right lane '''
    # Take a histogram of the bottom half of the image
    histogram = np.sum(image[image.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((image, image, image))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(image.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 90 
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window+1)*window_height
        win_y_high = image.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return leftx_base,rightx_base,left_fit,right_fit,left_fitx,right_fitx,ploty,out_img

def curvature_radius_meter(trans, ploty, left_fitx, right_fitx ):
    '''calculate the radius of curvature by meter'''
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    radi = [left_curverad, right_curverad]
    curvature_string = "Radius of Curvature: " + str(int(radi[0])) + ", " + str(int(radi[1]))
    return curvature_string
   
def curvature_radius_pixel(trans, left_fit, right_fit):
    '''calculate the radius of curvature by pixel'''
    y_eval = np.max(trans[0])
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    radi = [left_curverad, right_curverad]
    curvature_string = "Radius of Curvature: " + str(int(radi[0])) + ", " + str(int(radi[1]))
    return curvature_string

def pos_from_center_meter(trans, leftx_base, rightx_base):
    '''calculate the distance between Vehicle and Center by meter '''
    pos = trans.shape[1]/2
    xm_per_pix = 3.7/700
    offset = abs(pos - (leftx_base + rightx_base)/2)*xm_per_pix
    location_string = "Vehicle Dist. from Center: " + str(round(offset,2)) +" m"
    return location_string 

def pos_from_center_pixel(trans, leftx_base, rightx_base):
    '''calculate the distance between Vehicle and Center by meter '''
    pos = trans.shape[1]/2
    offset = abs(pos - (leftx_base + rightx_base)/2)
    location_string = "Vehicle Dist. from Center: " + str(round(offset,2))+" pixels"
    return location_string 
def map_lane(image,origin_image,perspective_M,left_fitx,right_fitx,ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, np.matrix(perspective_M).I , (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    final_result = cv2.addWeighted(origin_image, 1, newwarp, 0.3, 0)
    return final_result
def final_image(image,leftx_base, left_fit,rightx_base, right_fit,ploty):
    curvature_string = curvature_radius_meter(image,ploty, left_fit, right_fit)
    location_string = pos_from_center_meter(image, leftx_base, rightx_base)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,curvature_string,(400,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(image,location_string,(400,100), font, 1,(255,255,255),2,cv2.LINE_AA)
    return image

