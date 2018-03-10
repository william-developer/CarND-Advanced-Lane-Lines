

## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort.png "Undistorted"
[image2]: ./output_images/test1.png "Road Transformed"
[image3]: ./output_images/binary_combo_example.png "Binary Example"
[image4]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image5]: ./output_images/color_fit_lines.png "Fit Visual"
[image6]: ./output_images/example_output.png "Output"
[video1]: ./test_videos_output/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `camera_calibrator.py`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then use the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I apply this distortion correction to the test image using the `cv2.undistort()` function and obtain this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I use a combination of color and gradient thresholds to generate a binary image called `combine_gradient()` (thresholding steps at lines 90 through 105 in `lane_functions.py`).  
The first step, I take the derivative in x by function `cv2.Sobel`,and set threshold x gradient as (20, 100).
```python
    #Convert to grayscale
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
```
The second step, I convert image to HLS color space and separate the S channel and L channel,and setting the threshold color of S channel as (170, 255) and L channel as (100,255).
```python
    #hls color thresholds
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel=='s':
        s_channel = hls[:,:,2]
    elif channel=='h':
        s_channel = hls[:,:,0]
    elif channel=='l':
        s_channel = hls[:,:,1]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
```
The last step,I combine the three binary thresholds.
```python
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1)&(l_binary ==1)) | (sxbinary == 1)] = 1
```

Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_perspective()`, which appears in lines 107 through 126 in the file `lane_functions.py`.  The `warp_perspective()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 183, 0        | 
| 203, 720      | 183, 720      |
| 1127, 720     | 1024, 720     |
| 720, 460      | 1024, 0       |

I verify that my perspective transform is working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
Firstly,I find starting point by using peak position of histogram.
```python
     # Take a histogram of the bottom half of the image
     histogram = np.sum(image[image.shape[0]//2:,:], axis=0)
     # Create an output image to draw on and  visualize the result
     out_img = np.dstack((image, image, image))*255
     # Find the peak of the left and right halves of the histogram
     # These will be the starting point for the left and right lines
     midpoint = np.int(histogram.shape[0]/2)
     leftx_base = np.argmax(histogram[:midpoint])
     rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```
Secondly,I find all points through the windows one by one.(lines 165 through 190)
```python
   for window in range(nwindows):
   ...
```

At last,I do some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:
```python
     left_fit = np.polyfit(lefty, leftx, 2)
     right_fit = np.polyfit(righty, rightx, 2)
```


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculate the radius of curvature of the lane in lines 210 through 227 in my code in `lane_functions.py` in the function `curvature_radius_meter()`.
The first step,I convert pixels to meters,then fit my lane lines with a 2nd order polynomial.
```python
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
```
The second step,I calculate the radius of curvature of the lane.
```python
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left    _fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*r    ight_fit_cr[0])
```
I calculate the position of the vehicle with respect to center. in lines 238 through 244 in my code in `lane_functions.py` in the function `pos_from_center_meter()`.
I calculate pixels between the vehicle and lane center,then I converte pixels to meters.
```python
    pos = trans.shape[1]/2
    xm_per_pix = 3.7/700
    offset = abs(pos - (leftx_base + rightx_base)/2)*xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemente this step in lines 252 through 268 in my code in `lane_functions.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

When converting to a bird's eye view, I need to make 4 points of the original image, and then find the 4 points of the bird's eye view, which is more difficult. It is necessary to adjust the parameters for a long time, and the influence of the parameters on the transform matrix is large, and the transform matrix can affect the location of the lane. Moreover, the image quality of each frame is different. The image with low quality (the lane was not clear) could not locate the lane or find the lane accurately when making aerial view. In this scenario, the program could not work well. In the actual scene, the image of each frame may have a situation that lane blurred . However, it is good to use the characteristic of frame which is continuous, and the curvature deviation between successive frames is very small. When the lane is unclear, the mean lane lines of the few preceding frames can be fine. It is more robust, and the result will be better.

