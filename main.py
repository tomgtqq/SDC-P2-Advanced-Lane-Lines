import glob
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


class LaneFinding:
    """
    The class find the lane line
    """

    def __init__(self, g_thresh=(30, 255), m_thresh=(40, 255), d_thresh=(0, np.pi / 8),
                 sobel_kernel=3, show_birdeye=True, s_thresh=(90, 255), l_thresh=(150, 200),
                 b_thresh=(225, 255), lines_max_width=450):

        # Gradient threshold
        self.g_thresh = g_thresh
        self.m_thresh = m_thresh
        self.d_thresh = d_thresh
        self.sobel_kernel = sobel_kernel
        self.lines_max_width = lines_max_width

        # Color select
        self.s_thresh = s_thresh
        self.l_thresh = l_thresh
        self.b_thresh = b_thresh

        # Perspective Transform
        self.show_birdeye = show_birdeye
        self.birdseye_view_pickle = {}

        # Camera Calibration
        dist_pickle = pickle.load(open("pickle/result_dist_mtx_pickle", "rb"))
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]

        # Process image
        self.frame = None
        self.color_binary = None
        self.combine_binary = None
        self.img_warped = None
        self.out_img = None

        self.pre_left_fitx = None
        self.pre_right_fitx = None
        self.pre_ploty = None

    def __cal_perspective_matrix(self):
        """
        Calculate the perspective matrix
        """
        height, width = self.frame.shape[:2]
        offset = 450

        # source coordinates
        src = np.float32([[575, 465], [705, 465],
                          [255, 685], [1050, 685]])

        # desired coordinates
        dst = np.float32([[offset, 0], [width - offset, 0],
                          [offset, height], [width - offset, height]])

        # compute the perspective transform M given source and destination points
        M = cv2.getPerspectiveTransform(src, dst)
        # Compute the inverse perspective transform
        Minv = cv2.getPerspectiveTransform(dst, src)

        self.birdseye_view_pickle["M"] = M
        self.birdseye_view_pickle["Minv"] = Minv

    def __color_select(self):
        """
        Select color by SLB & LUV & LAB
        Returns:
           combined_binary (np array): binary image with combined threshod
        """
        img = self.frame
        s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
        l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:, :, 0]
        b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, 2]

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 1

        b_thresh_min = 155
        b_thresh_max = 200
        b_binary = np.zeros_like(b_channel)
        b_binary[(b_channel >= self.l_thresh[0]) & (b_channel <= self.l_thresh[1])] = 1

        l_thresh_min = 225
        l_thresh_max = 255
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= self.b_thresh[0]) & (l_channel <= self.b_thresh[1])] = 1

        combined_binary = np.zeros_like(s_binary)
        combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

        return combined_binary

    def __dir_threshold(self):
        """
        Calculate direction gradient
        Returns:
           binary_output (np array): binary image with direction gradient threshod
        """
        img = self.color_binary
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)

        # Calculate the gradient direciton
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= self.d_thresh[0]) & (absgraddir <= self.d_thresh[1])] = 1

        return binary_output

    def __mag_threshold(self):
        """
        Calculate the magnitude of the gradient
        Returns:
           binary_output (np array): Image binary matrix
        """
        img = self.color_binary

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)

        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)

        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= self.m_thresh[0]) & (gradmag <= self.m_thresh[1])] = 1
        return binary_output

    def __abs_sobel_thresh(self, orient='x'):
        """
        Calculate the magnitude of the gradient
        Args:
           orient(char): Sobel orientation
        Returns:
           binary_output(np arry): binary image matrix
        """

        img = self.color_binary
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if orient == 'x':
            abs_soebl = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel))
        if orient == 'y':
            abs_soebl = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel))

        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_soebl / np.max(abs_soebl))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= self.g_thresh[0]) & (scaled_sobel <= self.g_thresh[1])] = 1
        return binary_output

    def __combine_thresholds(self):
        """
        Combine gradient threshold & magnitude of the gradient
        Args:
           orient(char): Sobel orientation
        Returns:
           binary_output(np arry): binary image matrix
        """
        # Calculate the x and y gradient
        gradx = self.__abs_sobel_thresh(orient='x')
        grady = self.__abs_sobel_thresh(orient='y')

        # Calculate the magnitude of the gradient
        mag_binary = self.__mag_threshold()

        # Calculate direction
        dir_binary = self.__dir_threshold()

        # Combine thresholds
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        return combined

    def __convert_perspective(self):
        """
        Perspective Transform
        Returns:
           warped(np arry): binary image matrix with birdeye view
        """
        dist_pickle = pickle.load(open("pickle/result_dist_mtx_pickle", "rb"))
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]

        M = self.birdseye_view_pickle["M"]
        img = self.combine_binary
        # undistort using mtx and dist
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        img_size = (undist.shape[1], undist.shape[0])

        # Warp an image using the perspective transform, M:
        warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)

        return warped

    def __process_image(self):
        """
            Apply camera calibration and distortion correction
            color & gradient threshold then get perspective transform image
        """
        self.__cal_perspective_matrix()
        # Select color by SLB & LUV & LAB
        img_binary = self.__color_select()

        self.color_binary = np.dstack((img_binary, img_binary, img_binary)) * 255
        # Gradient threshold & Magnitude of the Gradient
        self.combine_binary = self.__combine_thresholds()
        # Perspective Transform
        self.img_warped = self.__convert_perspective()

    def __find_lane_pixels(self):
        """
        Perspective Transform
        Returns:
           warped(np arry): binary image matrix with birdeye view
        """

        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.img_warped[self.img_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((self.img_warped, self.img_warped, self.img_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 10
        # Set the width of the windows +/- margin
        margin = 80
        # Set minimum number of pixels found to recenter window
        minpix = 40

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(self.img_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.img_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.img_warped.shape[0] - (window + 1) * window_height
            win_y_high = self.img_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        # Get index where save pixel value in array
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        # Get pixels x y
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def __sliding_windows_hist(self):
        # Check is no pixels
        if self.img_warped is None:
            print('__sliding_windows_hist self.img_warped is none')
            return None

        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.__find_lane_pixels()

        if leftx.any() == 0 or lefty.any() == 0 or rightx.any() == 0 or righty.any() == 0:
            print(f'Not found lane pixels')
            return None

        # Fit a second order polynomial to each using `np.polyfit`

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, self.img_warped.shape[0] - 1, self.img_warped.shape[0])
        try:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        if (right_fitx[-1] - left_fitx[-1]) > self.lines_max_width:
            print(f'Found wrong lane line {right_fitx[-1] - left_fitx[-1]}')
            return None

            # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        return out_img, left_fitx, right_fitx, ploty

    def __fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

        # while normally calculate a y-value  for a given x, here we do the opposite,
        # why? because we expect our lane lines to be mostly vertically-oriented.
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        return left_fitx, right_fitx, ploty

    def __search_around_poly(self):
        # Check is no pixels
        if self.img_warped is None:
            return None

        margin = 80

        left_fitx = self.pre_left_fitx
        right_fitx = self.pre_right_fitx
        ploty = self.pre_ploty

        if (right_fitx[-1] - left_fitx[-1]) > self.lines_max_width:
            print(f'Search_around_poly Found wrong lane line {right_fitx[-1] - left_fitx[-1]}')
            return None

        # Grab activated pixels
        nonzero = self.img_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_fit = np.polyfit(ploty, left_fitx, 2)
        right_fit = np.polyfit(ploty, right_fitx, 2)

        # within the +/- margin of our polynomial function
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                                                                 2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
                                                                                   2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if leftx.any() == 0 or lefty.any() == 0 or rightx.any() == 0 or righty.any() == 0:
            print(f'search_around_poly Not found lane pixels')
            return None

        left_fitx, right_fitx, ploty = self.__fit_poly(self.img_warped.shape, leftx, lefty, rightx, righty)

        # Visualization
        # Create an image to draw on and an image to show the selection window
        self.out_img = np.dstack((self.img_warped, self.img_warped, self.img_warped)) * 255
        window_img = np.zeros_like(self.out_img)
        # Color in left and right line pixels
        self.out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        self.out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(self.out_img, 1, window_img, 0.3, 0)

        return result, left_fitx, right_fitx, ploty

    def __measure_curvature_real(self, left_fitx, right_fitx, ploty):
        """
        Calculates the curvature of polynomial functions in meters.
        """
        # Choose point to compute curvature just in front of the car
        yvalue = self.frame.shape[0]

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Pixels to real-world
        left_fit_converted = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_converted = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

        # Compute curvature radius
        left_curv_radius = ((1 + (2 * left_fit_converted[0] * yvalue + left_fit_converted[1]) ** 2) ** 1.5) / (
                2 * np.absolute(left_fit_converted[0]))
        right_curv_radius = ((1 + (2 * right_fit_converted[0] * yvalue + right_fit_converted[1]) ** 2) ** 1.5) / (
                2 * np.absolute(right_fit_converted[0]))

        # Compute distance in meters of vehicle center from the line
        car_center = self.frame.shape[1] / 2  # we assume the camera is centered in the car
        lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
        center_dist = (lane_center - car_center) * xm_per_pix

        return left_curv_radius, right_curv_radius, center_dist

    def __draw_detected_lane(self, left_fitx, right_fitx, ploty):
        """
        Draw the lane lines and information of car position
        Args:
           left_fitx(np array): fit x pixels with left line
           right_fitx(np array): fit x pixels with right line
           ploty(np array): Sobel fit y pixels with lines
        Returns:
           lane_img(np arry): image matrix with drawn data
        """
        # Read Minv matrix
        Minv = self.birdseye_view_pickle["Minv"]

        if Minv is None:
            self.__cal_perspective_matrix()

        img_copy = np.copy(self.frame)
        height, width = img_copy.shape[:2]

        # # create a canvas
        warp_zero = np.zeros_like(img_copy).astype(np.uint8)

        # transform for vc2 fillpoly
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # fillpoly lane line
        cv2.fillPoly(warp_zero, np.int_([pts]), (0, 255, 0))
        cv2.polylines(warp_zero, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=20)
        cv2.polylines(warp_zero, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=20)

        # unwarp lane line, Then add to lane img
        unwarp_img = cv2.warpPerspective(warp_zero, Minv, (width, height))
        lane_img = cv2.addWeighted(img_copy, 1, unwarp_img, 0.3, 0)

        # get lane lines information
        left_curverad, right_curverad, center = self.__measure_curvature_real(left_fitx, right_fitx, ploty)

        # draw lane lines information
        text_radius = 'Radius of Curvature is {} (m)'.format((left_curverad + right_curverad) // 2)
        cv2.putText(lane_img, text_radius, (50, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0),
                    thickness=2)
        text_center = 'Vehicle is center'

        if center > 0:
            text_center = 'Vehicle is left {:.2f} (m) of center'.format(abs(center))

        if center < 0:
            text_center = 'Vehicle is right {:.2f} (m) of center'.format(abs(center))

        cv2.putText(lane_img, text_center, (50, 100), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0),
                    thickness=2)

        return lane_img

    def process_video_pipline(self, frame):
        """
        process pipline
        Args:
           frame(np array:RGB): image matrix
        Returns:
           lane_img(np arry): image matrix with drawn data
        """

        # Coverted to BGR format image
        self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Apply camera calibration and distortion correction
        # color & gradient threshold then get perspective transform image
        self.__process_image()

        # Check lane lines pixels
        if self.img_warped is None:
            print('Check lane lines pixels is None')
            return frame

        # Find lane lines using sliding windows histgrame
        if self.pre_left_fitx is None or self.pre_right_fitx is None:
            if self.__sliding_windows_hist() is None:
                print('sliding_windows_hist is None')
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_img, left_fitx, right_fitx, ploty = self.__sliding_windows_hist()
        else:
            if self.__search_around_poly() is None:
                print('search_around_poly is None')
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Search from prior
            out_img, left_fitx, right_fitx, ploty = self.__search_around_poly()

        self.pre_left_fitx = left_fitx
        self.pre_right_fitx = right_fitx
        self.pre_ploty = ploty
        self.out_img = out_img
        # Draw lane lines and information
        lane_img = self.__draw_detected_lane(left_fitx, right_fitx, ploty)
        # Drawn the birdeye view
        if self.show_birdeye:
            # Add transformed images to the original image
            result = np.copy(lane_img)
            img_1 = cv2.resize(out_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

            # Find where to show birdeye view
            offset = 50
            endy = offset + img_1.shape[0]
            endx_img_1 = self.frame.shape[1] - offset
            startx_img_1 = endx_img_1 - img_1.shape[1]

            # Fill pixels of birdeye view image
            result[offset:endy, startx_img_1:endx_img_1] = img_1

            return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        return cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d corners in real world space, object points
    imgpoints = []  # 2d corners in image plane, image points

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    operator_video = LaneFinding()
    video_output = 'output_result/video_output.mp4'
    clip1 = VideoFileClip("test_video/project_video.mp4")
    video_clip = clip1.fl_image(operator_video.process_video_pipline)
    video_clip.write_videofile(video_output, audio=False)
