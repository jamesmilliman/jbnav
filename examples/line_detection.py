import os
import cv2
import numpy as np

import functools


@functools.lru_cache(maxsize=None)
def get_camera_coeffs(cal_dir="camera_cal", corner_dims=(9, 6)):
    cal_files, imgs, gray_imgs = get_images_from_dir(cal_dir)

    objpoints = []
    imgpoints = []

    objp = np.zeros((corner_dims[0] * corner_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : corner_dims[0], 0 : corner_dims[1]].T.reshape(-1, 2)

    for img, filename in zip(gray_imgs, cal_files):
        # print(filename)
        ret, corners = cv2.findChessboardCorners(img, corner_dims, None)

        if ret == True:
            # new_img = img.copy()
            # cv2.drawChessboardCorners(new_img, corner_dims, corners, ret)
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            pass
            # print("ERROR")

    return cv2.calibrateCamera(
        objpoints, imgpoints, gray_imgs[0].shape[::-1], None, None
    )


def calibrate_imgs(imgs):
    ret, mtx, dist, rvecs, tvecs = get_camera_coeffs()

    return list(map(lambda img: cv2.undistort(img, mtx, dist, None, mtx), imgs))


def calibrate_img(img):
    ret, mtx, dist, rvecs, tvecs = get_camera_coeffs()

    return cv2.undistort(img, mtx, dist, None, mtx)


def sobel(img, thresh_mag=50, thresh_dir=(0.5, 1.5)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255 * sobel / np.max(sobel))

    dir_sobel = np.arctan2(np.abs(sobelx), np.abs(sobely))
    return (
        (scaled_sobel > thresh_mag)
        & (dir_sobel > thresh_dir[0])
        & (dir_sobel < thresh_dir[1])
    ).astype(np.uint8)

    # return (scaled_sobel > thresh_mag).astype(np.uint8)


def extract_channel(
    image, colorspace=cv2.COLOR_RGB2HLS, thresholds=(50, 100), channel=2
):
    converted_image = cv2.cvtColor(image, colorspace)
    extracted_channel = converted_image[:, :, channel]
    binary = np.zeros_like(extracted_channel)
    binary[
        (extracted_channel >= thresholds[0]) & (extracted_channel <= thresholds[1])
    ] = 1
    return binary


def line_detection(img, gray):
    s_channel = extract_channel(img, channel=2)
    l_channel = extract_channel(img, channel=1)

    s = sobel(gray)
    stacked_binary = np.zeros_like(s_channel)
    stacked_binary[(((s_channel == 1) & (l_channel == 1))) | (s == 1)] = 1
    return stacked_binary


def perspective_transform(image):
    imshape = image.shape
    height = imshape[0]
    width = imshape[1]
    width_ratio = 0.457
    width_ratio = 0.4
    height_ratio = 0.625
    height_ratio = 0.58
    top_left = (width * width_ratio, height * height_ratio)
    top_right = (width * (1 - width_ratio), height * height_ratio)
    bottom_left = (100, height)
    bottom_right = (width - 100, height)
    # print(top_left, top_right, bottom_left, bottom_right)

    src_vertices = np.float32([top_left, top_right, bottom_left, bottom_right])
    d1 = [100, 0]
    d2 = [width - 100, 0]
    d3 = [100, height]
    d4 = [width - 100, height]
    dst_vertices = np.float32([d1, d2, d3, d4])
    matrix = cv2.getPerspectiveTransform(src_vertices, dst_vertices)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    # We also calculate the oposite transform as we'll need it later
    unwarped = cv2.getPerspectiveTransform(dst_vertices, src_vertices)
    # Return the resulting image and matrix
    return warped, unwarped


def image_histogram(warped):
    return np.sum(warped, axis=0)


def sliding_window_search(
    wrapped_image, draw_sliding_windows=False, visualize_detected_lanes=False
):
    out_img = np.dstack((wrapped_image, wrapped_image, wrapped_image)) * 255
    histo = image_histogram(wrapped_image)
    midpoint = np.int(histo.shape[0] / 2)
    leftx_base = np.argmax(histo[:midpoint])
    rightx_base = np.argmax(histo[midpoint:]) + midpoint
    num_windows = 9
    window_height = np.int(wrapped_image.shape[0] / num_windows)
    nonzero = wrapped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100
    minpix = 50
    left_lane_indexes = []
    right_lane_indexes = []

    for window in range(num_windows):
        # Identify window boundaries in x and y (and right and left)
        top_y = wrapped_image.shape[0] - (window + 1) * window_height
        bottom_y = wrapped_image.shape[0] - window * window_height
        left_lane_leftx = leftx_current - margin
        left_lane_rightx = leftx_current + margin
        right_lane_leftx = rightx_current - margin
        right_lane_rightx = rightx_current + margin

        if draw_sliding_windows:
            cv2.rectangle(
                out_img,
                (left_lane_leftx, top_y),
                (left_lane_rightx, bottom_y),
                (0, 255, 0),
                2,
            )
            cv2.rectangle(
                out_img,
                (right_lane_leftx, top_y),
                (right_lane_rightx, bottom_y),
                (0, 255, 0),
                2,
            )

        # Identify the nonzero pixels in x and y within the window
        good_left_indexes = (
            (nonzeroy >= top_y)
            & (nonzeroy < bottom_y)
            & (nonzerox >= left_lane_leftx)
            & (nonzerox < left_lane_rightx)
        ).nonzero()[0]
        good_right_indexes = (
            (nonzeroy >= top_y)
            & (nonzeroy < bottom_y)
            & (nonzerox >= right_lane_leftx)
            & (nonzerox < right_lane_rightx)
        ).nonzero()[0]

        left_lane_indexes.append(good_left_indexes)
        right_lane_indexes.append(good_right_indexes)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_indexes) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_indexes]))
        if len(good_right_indexes) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_indexes]))

    # Concatenate the arrays of indices
    left_lane_indexes = np.concatenate(left_lane_indexes)
    right_lane_indexes = np.concatenate(right_lane_indexes)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_indexes]
    lefty = nonzeroy[left_lane_indexes]
    rightx = nonzerox[right_lane_indexes]
    righty = nonzeroy[right_lane_indexes]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    out_img[nonzeroy[left_lane_indexes], nonzerox[left_lane_indexes]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_indexes], nonzerox[right_lane_indexes]] = [0, 0, 255]
    ploty = np.linspace(0, wrapped_image.shape[0] - 1, wrapped_image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    window_img = np.zeros_like(out_img)
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))]
    )
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([right_fitx - margin, ploty]))]
    )
    right_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))]
    )
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    if visualize_detected_lanes:
        plt.figure(figsize=(20, 10))
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color="yellow")
        plt.plot(right_fitx, ploty, color="yellow")
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return left_fit, right_fit


def calculate_curve_radius(wrapped_image, left_fit, right_fit):
    ym_per_pix = 30 / wrapped_image.shape[0] * 0.625  # meters per pixel in y dimension
    xm_per_pix = 3.7 / wrapped_image.shape[1] * 0.7  # meters per pixel in x dimension

    ploty = np.linspace(0, wrapped_image.shape[0] - 1, wrapped_image.shape[0])
    leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    car_position = wrapped_image.shape[1] // 2

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    max_y = np.max(ploty)

    # _fit is a 2 degree polynomial of the form y = ax^2 + bx + c
    # From wiki: https://en.wikipedia.org/wiki/Radius_of_curvature#In_2D
    left_first_deriv = 2 * left_fit_cr[0] + left_fit_cr[1]
    left_secnd_deriv = 2 * left_fit_cr[0]
    left_radius = int(
        ((1 + (left_first_deriv ** 2) ** 1.5) / np.absolute(left_secnd_deriv))
    )

    right_first_deriv = 2 * right_fit_cr[0] + right_fit_cr[1]
    right_secnd_deriv = 2 * right_fit[0]
    right_radius = int(
        ((1 + (right_first_deriv ** 2) ** 1.5) / np.absolute(right_secnd_deriv))
    )

    left_lane_bottom = (left_fit[0] * max_y) ** 2 + left_fit[1] * max_y + left_fit[2]
    right_lane_bottom = (
        (right_fit[0] * max_y) ** 2 + right_fit[1] * max_y + right_fit[2]
    )

    actual_position = (left_lane_bottom + right_lane_bottom) / 2

    distance = np.absolute((car_position - actual_position) * xm_per_pix)

    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')
    return (left_radius + right_radius) / 2, distance
    # Example values: 632.1 m    626.2 m


def draw_unwrapped(original_image, wrapped_image, unwrapped_image, left_fit, right_fit):
    height, width, _ = wrapped_image.shape
    ploty = np.linspace(0, wrapped_image.shape[0] - 1, wrapped_image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    warp_zero = np.zeros_like(wrapped_image).astype(np.uint8)
    color_warp = np.zeros_like(wrapped_image).astype(np.uint8)

    ploty = np.linspace(0, height - 1, num=height)  # to cover same y-range as image

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    points = np.int_([pts])

    cv2.fillPoly(color_warp, points, (0, 255, 0, 0))
    cv2.polylines(
        color_warp,
        np.int32([pts_left]),
        isClosed=False,
        color=(255, 0, 255),
        thickness=15,
    )
    cv2.polylines(
        color_warp,
        np.int32([pts_right]),
        isClosed=False,
        color=(0, 255, 255),
        thickness=15,
    )

    # Warp the blank back to original image space using inverse perspective matrix (unwrapped_image)
    newwarp = cv2.warpPerspective(color_warp, unwrapped_image, (width, height))

    result = cv2.addWeighted(original_image, 1, newwarp, 0.5, 0)
    radius, distance = calculate_curve_radius(wrapped_image, left_fit, right_fit)
    cv2.putText(
        result,
        "Radius of curve is " + str(int(radius)) + "m",
        (100, 100),
        2,
        1,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        result,
        "Distance from center is {:2f}".format(distance) + "m",
        (100, 150),
        2,
        1,
        (255, 255, 0),
        2,
    )
    return result


def lines(img):
    try:
        orig_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dst = img
        # dst = calibrate_img(img)
        # gray = calibrate_img(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lines = line_detection(img, gray)
        return lines
    except:
        return orig_img, None
