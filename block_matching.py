import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pickle


def left_block_match(left_img, right_img, window):
    # convert images to black and white
    left = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
    right = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

    # store the height and width of the matrix represented left image
    h, w = left.shape

    # create matrix the same size as left image to store disparity values
    DL = np.zeros((h, w), np.uint8)

    # determine the radius of the window
    window_rad = window//2

    # pad the sides of both images with the window size
    left_padded = cv.copyMakeBorder(left, window_rad, window_rad, window_rad, window_rad, cv.BORDER_REPLICATE)
    right_padded = cv.copyMakeBorder(right, window_rad, window_rad, window_rad, window_rad, cv.BORDER_REPLICATE)

    # for each row in left matrix
    for i in range(0, h):
        # for each column in left matrix
        for j in range(0, w):

            # print runtime progress
            print(i, j)

            # extract a block from left image
            left_block = left_padded[i:i + window, j:j + window]

            # initialize disparity as 0 and minimum sum of squared distance as a large value
            disparity = 0
            min_ssd = 100000

            # for each column in right matrix
            for jr in range(0, w):

                # store the disparity we are testing
                try_disparity = abs(jr - j)

                # extract a block from the right image
                right_block = right_padded[i:i + window, jr:jr + window]

                # find SSD between two blocks
                diffs = abs(left_block - right_block)
                squared_diffs = np.multiply(diffs, diffs)
                ssd = np.sum(squared_diffs)

                # if this SSD value is the smallest tested so far,
                # update minimum SSD and store this disparity value in disparity
                if ssd < min_ssd:
                    min_ssd = ssd
                    disparity = try_disparity

            # update the i, j position of disparity matrix with the disparity of the smallest SSD
            DL[i, j] = disparity
    return DL


def right_block_match(left_img, right_img, window):
    # convert images to black and white
    left = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
    right = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

    # store the height and width of the matrix represented right image
    h, w = right.shape

    # create matrix the same size as left image to store disparity values
    DR = np.zeros((h, w), np.uint8)

    # determine the radius of the window
    window_rad = window // 2

    # pad the sides of both images with the window size
    left_padded = cv.copyMakeBorder(left, window_rad, window_rad, window_rad, window_rad, cv.BORDER_REPLICATE)
    right_padded = cv.copyMakeBorder(right, window_rad, window_rad, window_rad, window_rad, cv.BORDER_REPLICATE)

    # for each row in right matrix
    for i in range(0, h):
        # for each column in right matrix
        for j in range(0, w):

            # print runtime progress
            print(i, j)

            # extract a block from right image
            right_block = right_padded[i:i + window, j:j + window]

            # initialize disparity as 0 and minimum sum of squared distance as a large value
            disparity = 0
            min_ssd = 100000

            # for each column in left matrix
            for jl in range(0, w):

                # store the disparity we are testing
                try_disparity = abs(jl - j)

                # extract a block from the left image
                left_block = left_padded[i:i + window, jl:jl + window]

                # find SSD between two blocks
                diffs = abs(right_block - left_block)
                squared_diffs = np.multiply(diffs, diffs)
                ssd = np.sum(squared_diffs)

                # if this SSD value is the smallest tested so far,
                # update minimum SSD and store this disparity value in disparity
                if ssd < min_ssd:
                    min_ssd = ssd
                    disparity = try_disparity

            # update the i, j position of disparity matrix with the disparity of the smallest SSD
            DR[i, j] = disparity
    return DR


def opt_left_block_match(left_img, right_img, window):
    # convert images to black and white
    left = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
    right = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

    # store the height and width of the matrix represented left image
    h, w = left.shape

    # create matrix the same size as left image to store disparity values
    DL = np.zeros((h, w), np.uint8)

    # determine the radius of the window
    window_rad = window//2

    # pad the sides of both images with the window size
    left_padded = cv.copyMakeBorder(left, window_rad, window_rad, window_rad, window_rad, cv.BORDER_REPLICATE)
    right_padded = cv.copyMakeBorder(right, window_rad, window_rad, window_rad, window_rad, cv.BORDER_REPLICATE)

    # for each row in left matrix
    for i in range(0, h):
        # for each column in left matrix
        for j in range(0, w):

            # print runtime progress
            print(i, j)

            # extract a block from left image
            left_block = left_padded[i:i + window, j:j + window]

            # initialize disparity as 0 and minimum sum of squared distance as a large value
            disparity = 0
            min_ssd = 100000

            # set maximum disparity value
            if j < 75:
                max_disp = j
            else:
                max_disp = 75

            matching_disparity = 0

            # for each column in right matrix
            for disparity in range(0, max_disp):

                # extract a block from the right image to the left of the left coordinates
                right_block = right_padded[i:i + window, j - disparity:j - disparity + window]

                # find SSD between two blocks
                diffs = abs(left_block - right_block)
                squared_diffs = np.multiply(diffs, diffs)
                ssd = np.sum(squared_diffs)

                # if this SSD value is the smallest tested so far,
                # update minimum SSD and store this disparity value in disparity
                if ssd < min_ssd:
                    min_ssd = ssd
                    matching_disparity = disparity

            # update the i, j position of disparity matrix with the disparity of the smallest SSD
            DL[i, j] = matching_disparity
    return DL


def opt_right_block_match(left_img, right_img, window):
    # convert images to black and white
    left = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
    right = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

    # store the height and width of the matrix represented right image
    h, w = right.shape

    # create matrix the same size as right image to store disparity values
    DR = np.zeros((h, w), np.uint8)

    # determine the radius of the window
    window_rad = window // 2

    # pad the sides of both images with the window size
    left_padded = cv.copyMakeBorder(left, window_rad, window_rad, window_rad, window_rad, cv.BORDER_REPLICATE)
    right_padded = cv.copyMakeBorder(right, window_rad, window_rad, window_rad, window_rad, cv.BORDER_REPLICATE)

    # for each row in right matrix
    for i in range(0, h):
        # for each column in right matrix
        for j in range(0, w):

            # print runtime progress
            print(i, j)

            # extract a block from right image
            right_block = right_padded[i:i + window, j:j + window]

            # initialize disparity as 0 and minimum sum of squared distance as a large value
            disparity = 0
            min_ssd = 100000

            # set maximum disparity value
            if w - j < 75:
                max_disp = w - j
            else:
                max_disp = 75

            matching_disparity = 0

            # for each column in left matrix
            for disparity in range(0, max_disp):

                # extract a block from the left image to the right of the right coordinates
                left_block = left_padded[i:i + window, j + disparity:j + disparity + window]

                # find SSD between two blocks
                diffs = abs(left_block - right_block)
                squared_diffs = np.multiply(diffs, diffs)
                ssd = np.sum(squared_diffs)

                # if this SSD value is the smallest tested so far,
                # update minimum SSD and store this disparity value in disparity
                if ssd < min_ssd:
                    min_ssd = ssd
                    matching_disparity = disparity

            # update the i, j position of disparity matrix with the disparity of the smallest SSD
            DR[i, j] = matching_disparity
    return DR


def consistency_check_left(left_disp_map, right_disp_map):
    h, w = left_disp_map.shape
    disparity = 0
    for i in range(0, h):
        for j in range(0, w):
            disparity = int(left_disp_map[i, j])
            if right_disp_map[i, j - disparity] != disparity:
               #  print("inconsistancy on left at", i, j)
                left_disp_map[i, j] = 0
    return left_disp_map


def consistency_check_right(left_disp_map, right_disp_map):
    h, w = right_disp_map.shape
    disparity = 0
    for i in range(0, h):
        for j in range(0, w):
            disparity = int(right_disp_map[i, j])
            if left_disp_map[i, j + disparity] != disparity:
                # print("inconsistency on right at", i, j)
                right_disp_map[i, j] = 0
    return right_disp_map


def calculate_mse(est_disp, gt_disp):
    h, w = est_disp.shape
    squared_errors = 0
    inconsistencies = 0
    for i in range(0, h):
        for j in range(0, w):
            if est_disp[i, j] == 0:
                inconsistencies += 1
            else:
                difference = int(est_disp[i, j]) - int(gt_disp[i, j])
                squared_errors += difference * difference
    mse = squared_errors/((h*w) - inconsistencies)
    return mse


# first part
left_image = cv.imread("view1.png")
right_image = cv.imread("view5.png")

# left_disparity_3 = left_block_match(left_image, right_image, 3)
# right_disparity_3 = right_block_match(left_image, right_image, 3)

# with open('left_disparity_3.pk', 'wb') as fi:
#     pickle.dump(left_disparity_3, fi)

# with open('right_disparity_3.pk', 'wb') as fi:
#     pickle.dump(right_disparity_3, fi)

with open('left_disparity_3.pk', 'rb') as fi:
    left_disparity_3 = pickle.load(fi)

with open('right_disparity_3.pk', 'rb') as fi:
    right_disparity_3 = pickle.load(fi)


# cv.imshow('left image', left_image)
# cv.imshow('right image', right_image)
# cv.imshow('left image disparity map', left_disparity_3)
# cv.imshow('right image disparity map', right_disparity_3)

# plt.imshow(right_disparity_3)
# plt.show()

# second part
# left_disparity_5 = left_block_match(left_image, right_image, 5)
# right_disparity_5 = right_block_match(left_image, right_image, 5)

# left_disparity_7 = left_block_match(left_image, right_image, 7)
# right_disparity_7 = right_block_match(left_image, right_image, 7)

# left_disparity_9 = left_block_match(left_image, right_image, 9)
# right_disparity_9 = right_block_match(left_image, right_image, 9)

# left_disparity_11 = left_block_match(left_image, right_image, 11)
# right_disparity_11 = right_block_match(left_image, right_image, 11)

# with open('left_disparity_5.pk', 'wb') as fi:
#     pickle.dump(left_disparity_5, fi)

# with open('right_disparity_5.pk', 'wb') as fi:
#     pickle.dump(right_disparity_5, fi)

# with open('left_disparity_7.pk', 'wb') as fi:
#     pickle.dump(left_disparity_7, fi)

# with open('right_disparity_7.pk', 'wb') as fi:
#     pickle.dump(right_disparity_7, fi)

# with open('left_disparity_9.pk', 'wb') as fi:
#     pickle.dump(left_disparity_9, fi)

# with open('right_disparity_9.pk', 'wb') as fi:
#     pickle.dump(right_disparity_9, fi)

# with open('left_disparity_11.pk', 'wb') as fi:
#     pickle.dump(left_disparity_11, fi)

# with open('right_disparity_11.pk', 'wb') as fi:
#     pickle.dump(right_disparity_11, fi)

with open('left_disparity_5.pk', 'rb') as fi:
    left_disparity_5 = pickle.load(fi)

with open('right_disparity_5.pk', 'rb') as fi:
    right_disparity_5 = pickle.load(fi)

with open('left_disparity_7.pk', 'rb') as fi:
    left_disparity_7 = pickle.load(fi)

with open('right_disparity_7.pk', 'rb') as fi:
    right_disparity_7 = pickle.load(fi)

with open('left_disparity_9.pk', 'rb') as fi:
    left_disparity_9 = pickle.load(fi)

with open('right_disparity_9.pk', 'rb') as fi:
    right_disparity_9 = pickle.load(fi)

with open('left_disparity_11.pk', 'rb') as fi:
    left_disparity_11 = pickle.load(fi)

with open('right_disparity_11.pk', 'rb') as fi:
    right_disparity_11 = pickle.load(fi)

gt_1_image = cv.imread("disp1.png")
gt_2_image = cv.imread("disp5.png")
gt_1 = cv.cvtColor(gt_1_image, cv.COLOR_BGR2GRAY)
gt_2 = cv.cvtColor(gt_2_image, cv.COLOR_BGR2GRAY)


# plot left disparity MSE
# bars_at = [1, 2, 3, 4, 5]
# left_mse = [calculate_mse(left_disparity_3, gt_1),
#             calculate_mse(left_disparity_5, gt_1),
#             calculate_mse(left_disparity_7, gt_1),
#             calculate_mse(left_disparity_9, gt_1),
#             calculate_mse(left_disparity_11, gt_1)]
# labels = ['3x3', '5x5', '7x7', '9x9', '11x11']
# plt.bar(bars_at, left_mse, tick_label=labels, width=0.75, color=['orange'])
# plt.xlabel('Window dimension used')
# plt.ylabel('Mean Square Error')
# plt.title('Left disparity MSE')
# plt.show()

# plot right disparity MSE
# right_mse = [calculate_mse(right_disparity_3, gt_2),
#              calculate_mse(right_disparity_5, gt_2),
#              calculate_mse(right_disparity_7, gt_2),
#              calculate_mse(right_disparity_9, gt_2),
#              calculate_mse(right_disparity_11, gt_2)]
# plt.bar(bars_at, right_mse, tick_label=labels, width=0.8, color=['green'])
# plt.xlabel('Window dimension used')
# plt.ylabel('Mean Square Error')
# plt.title('Right disparity MSE')
# plt.show()


# part 3
# opt_left_disp = opt_left_block_match(left_image, right_image, 3)
# opt_right_disp = opt_right_block_match(left_image, right_image, 3)

# with open('optimized_left_disparity.pk', 'wb') as fi:
#     pickle.dump(opt_left_disp, fi)

# with open('optimized_right_disparity.pk', 'wb') as fi:
#     pickle.dump(opt_right_disp, fi)

with open('optimized_left_disparity.pk', 'rb') as fi:
    opt_left_disp = pickle.load(fi)

with open('optimized_right_disparity.pk', 'rb') as fi:
    opt_right_disp = pickle.load(fi)
cv.imshow('optimized left image disparity map', opt_left_disp)
cv.imshow('optimized right image disparity map', opt_right_disp)

left_disp_const = consistency_check_left(opt_left_disp, opt_right_disp)
right_disp_const = consistency_check_right(opt_left_disp, opt_right_disp)
optimized_mse_left = calculate_mse(left_disparity_3, gt_1)
optimized_mse_right = calculate_mse(right_disparity_3, gt_2)
print(optimized_mse_left)
print(optimized_mse_right)

cv.imshow('left disparity after improvements', left_disp_const)
cv.imshow('right disparity after improvements', right_disp_const)
plt.imshow(right_disparity_3)
plt.show()





