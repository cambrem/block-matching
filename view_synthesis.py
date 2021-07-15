import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def synthesize():
    # loop through all pixels in left disparity map
    for i in range(h):
        for j in range(w):
            # initialize synthesized view values to zero (because syn_view is initially a copy of left view)
            syn_view[i, j] = 0
            # determine disparity and half disparity values at current pixel
            disparity_left = gt_1[i, j]
            half_disp_left = gt_1_half[i, j]
            # if the half disparity value will map back to a valid index in syn_view
            if j - half_disp_left >= 0:
                # mark the corresponding syn_view pixel with a pixel from the left view
                syn_view[i, j - half_disp_left] = left_image[i, j]
                # store the disparity value used to determine that pixel in syn_view at the same index
                resolved[i, j - half_disp_left] = disparity_left
                # turn that pixel white in filled_by_left
                filled_by_left[i, j - half_disp_left] = 255

    # loop through all pixels in right disparity map
    for i in range(h):
        for j in range(w):
            # determine disparity and half disparity values at current pixel
            disparity_right = gt_2[i, j]
            half_disp_right = gt_2_half[i, j]
            # if the half disparity will map back to a valid index in syn_view
            if j + half_disp_right < 463:
                # if the current disparity value is greater than the one currently being used
                if resolved[i, j + half_disp_right] < disparity_right:
                    # update the corresponding syn_view pixel with a pixel from the right view
                    syn_view[i, j + half_disp_right] = right_image[i, j]
                    # store the disparity value used to determine that pixel in syn_view at the same index
                    resolved[i, j + half_disp_right] = disparity_right
                    # turn that pixel white in filled_by_right, and turn it back to black in filled_by_left
                    filled_by_right[i, j + half_disp_right] = 255
                    filled_by_left[i, j + half_disp_right] = 0


# load images
left_image = cv.imread("view1.png")
right_image = cv.imread("view5.png")
gt_1_image = cv.imread("disp1.png")
gt_2_image = cv.imread("disp5.png")

# convert ground truth disparity maps to gray
gt_1 = cv.cvtColor(gt_1_image, cv.COLOR_BGR2GRAY)
gt_2 = cv.cvtColor(gt_2_image, cv.COLOR_BGR2GRAY)

# create matrices for right and left views with half disparity values
gt_1_half = gt_1 // 2
gt_2_half = gt_2 // 2

# determine the shape of the ground truth disparity maps
h, w = gt_1.shape

# create a copy of left image (bgr matrix) to store the synthesized view
syn_view = left_image.copy()

# create resolved matrix to store disparity values used to create synthesized views
resolved = np.zeros((h, w), np.uint8)

# create matrices to track which pixels are being taken from which view
filled_by_left = np.zeros((h, w), np.uint8)
filled_by_right = np.zeros((h, w), np.uint8)

# call the synthesize function and display results
synthesize()
cv.imshow('left view', left_image)
cv.imshow('right view', right_image)
cv.imshow('pixels filled in by left image', filled_by_left)
cv.imshow('pixels filled in by right image', filled_by_right)
cv.imshow('synthesized image', syn_view)
plt.imshow(syn_view)
plt.show()




