import cv2
import numpy as np
import os
import sys

# method for matching key points of descriptors using SSDR
# start:
def knnMatch(x1, x2):
    res = list()
    for i, x in enumerate(x1):
        x = np.tile(x, (x2.shape[0], 1))
        distances = np.linalg.norm(x - x2, axis = 1)
        k_idx = np.argsort(distances)[: 2]
        distances = distances[k_idx]
        if distances[0] < 0.7 * distances[1]:
            temp = dict()
            temp['distance'] = distances[0]
            temp['queryIdx'] = i
            temp['trainIdx'] = k_idx[0]
            res.append(temp)
        # limitimg the good maches to 500 to enhance performance
        if len(res) == 500:
            break
    return res
# end

# method for calculating homography using SVD
# start:
def calculateHomography(src, dst):
#     ref: http://ros-developer.com/2017/12/26/finding-homography-matrix-using-singular-value-decomposition-and-ransac-in-opencv-and-matlab/

    A = list()
    for i in range(len(src)):
        A.append([-src[i][0], -src[i][1], -1, 0, 0, 0, src[i][0] * dst[i][0], src[i][1] * dst[i][0], dst[i][0]])
        A.append([0, 0, 0, -src[i][0], -src[i][1], -1, src[i][0] * dst[i][1], src[i][1] * dst[i][1], dst[i][1]])

    u, s, v = np.linalg.svd(np.array(A))
    h = np.reshape(v[8], (3, 3))
    return h
# end

# method for finding inliers and homography matrix using RANSAC
# start:
def findHomography(src, dst):
    inliers = []
    H = None
    for i in range(500):
        # get 10 random points for calculating homography
        idx = np.random.permutation(len(src))[: 10]
        src1 = src[idx]
        dst1 = dst[idx]
        # call the homography function on those points
        h = calculateHomography(src1, dst1)
        _inliers = []

        for i, src_pt in enumerate(src):
            p1 = np.append(src_pt, 1) # changing src points into homogeneous coordinates
            p2 = np.append(dst[i], 1) # changing dst points into homogeneous coordinates
            _p1 = np.matmul(h, p1) # applying homography to the src points and getting their corresponding coordinates in dst image
            _p1 = _p1 / _p1[2]
            d = np.linalg.norm(p2 - _p1) # finding the error between the two points
            if d < 5:
                _inliers.append(i) # the point is inlier if the error is less than 5

        if len(_inliers) > len(inliers): # keeping the best homography matrix in 500 iterations
            inliers = _inliers
            H = h

        if len(inliers) > (len(src) * 0.7):
            break
    print ("Src size: ", len(src), "Max inliers: ", len(inliers))
    return H, inliers
# end

# method used to trim black edges around an image
# start:
def trim(frame):
    top = left = 0
    bottom, right = frame.shape[: 2]

    # trim top
    while not np.sum(frame[top]):
        top += 1
    # trim bottom
    while not np.sum(frame[bottom - 1]):
        bottom -= 1
    # trim left
    while not np.sum(frame[:, left]):
        left += 1
    # trim right
    while not np.sum(frame[:, right - 1]):
        right -= 1

    return frame[top : bottom - 1, left : right - 1]
#end

# method used to find the laplacian pyramids of an image
# start:
def find_laplacian_pyramids(img, n):
    # Gaussian Pyramid
    layer = img.copy()
    gaussian_pyr = [layer]
    for i in range(n):
        layer = cv2.pyrDown(layer)
        gaussian_pyr.append(layer)

    # Laplacian Pyramid
    laplacian_pyr = [gaussian_pyr[n - 1]]
    for i in reversed(range(1, n)):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize = size)
        laplacian = cv2.subtract(gaussian_pyr[i - 1], gaussian_expanded)
        laplacian_pyr.append(laplacian)

    return laplacian_pyr
# end

# method used to blend 2 images using laplacian pyramids reconstruction
# start:
def blendImages(img1, img2):
    layers = 6
    laplacian1 = find_laplacian_pyramids(img1, layers)
    laplacian2 = find_laplacian_pyramids(img2, layers)

    # Laplacian Pyramid of combined image
    combined_laplacian = []
    for img1_lap, img2_lap in zip(laplacian1, laplacian2):
        laplacian = np.hstack((img1_lap, img2_lap))
        combined_laplacian.append(laplacian)

    # Reconstructing image
    reconstructed_image = combined_laplacian[0]
    for i in range(1, layers):
        size_cl = (combined_laplacian[i].shape[1], combined_laplacian[i].shape[0])
        size_ri = (reconstructed_image.shape[1], reconstructed_image.shape[0])

        reconstructed_image = cv2.pyrUp(reconstructed_image)#, dstsize = size)

        # making the size of both the layers equal before adding them up.
        # start:
        # padding zeroes around width
        a, b = reconstructed_image.shape[1], combined_laplacian[i].shape[1]
        c1 = 0
        if a > b:
            c1 = a - b
            pad = c1 // 2
            zeroes = np.zeros((size_cl[1], pad, 3), dtype = np.uint8)
            combined_laplacian[i] = np.hstack((zeroes, combined_laplacian[i], zeroes))
            if c1 % 2 != 0:
                combined_laplacian[i] = np.hstack((combined_laplacian[i], np.zeros((size_cl[1], 1, 3), dtype = np.uint8)))
        elif b > a:
            c1 = b - a
            pad = c1 // 2
            zeroes = np.zeros((size_ri[1], pad, 3), dtype = np.uint8)
            reconstructed_image = np.hstack((zeroes, reconstructed_image, zeroes))
            if c1 % 2 != 0:
                reconstructed_image = np.hstack((reconstructed_image, np.zeros((size_ri[1], 1, 3), dtype = np.uint8)))

        # padding zeroes around height
        a, b = reconstructed_image.shape[0], combined_laplacian[i].shape[0]
        if a > b:
            c2 = a - b
            pad = c2 // 2
            zeroes = np.zeros((pad, size_cl[0] + c1, 3), dtype = np.uint8)
            combined_laplacian[i] = np.vstack((zeroes, combined_laplacian[i], zeroes))
            if c2 % 2 != 0:
                combined_laplacian[i] = np.vstack((combined_laplacian[i], np.zeros((1, size_cl[0] + c1, 3), dtype = np.uint8)))
        elif b > a:
            c2 = b - a
            pad = c2 // 2
            zeroes = np.zeros((pad, size_ri[0] + c1, 3), dtype = np.uint8)
            reconstructed_image = np.vstack((zeroes, reconstructed_image, zeroes))
            if c2 % 2 != 0:
                reconstructed_image = np.vstack((reconstructed_image, np.zeros((1, size_ri[0] + c1, 3), dtype = np.uint8)))
        # end
        reconstructed_image = cv2.add(combined_laplacian[i], reconstructed_image)
    return reconstructed_image
# end

# method used to stitch 2 images using warp perspective
# start:
def stitch(img1, img2, H, blend = False):
#     ref: https://github.com/pavanpn/Image-Stitching/blob/9b4f77684da56cff50051b364d366704078432ec/stitch_images.py#L6
    h1, w1 = img1.shape[: 2]
    h2, w2 = img2.shape[: 2]

    # Get the canvas dimesions
    dims1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    dims2_temp = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # Get relative perspective of second image
    dims2 = cv2.perspectiveTransform(dims2_temp, H)

    # Getting images together
    # Calculate dimensions of match points
    x_min, y_min = np.int32(np.amin(np.concatenate((dims1, dims2), axis = 0), axis = 0) - 0.5)[0]
    x_max, y_max = np.int32(np.amax(np.concatenate((dims1, dims2), axis = 0), axis = 0) + 0.5)[0]

    # Create output array after affine transformation
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    result_img = cv2.warpPerspective(img2, np.matmul(transform_array, H), (x_max - x_min, y_max - y_min))
    result_img[transform_dist[1] : h1 + transform_dist[1], transform_dist[0] : w1 + transform_dist[0]] = img1
    if result_img.shape != img1.shape and blend:
        if transform_dist[0] > 100:
            result_img = blendImages(result_img[:, : transform_dist[0]], result_img[:, transform_dist[0] : w1 + transform_dist[0]])
        else:
            result_img = blendImages(result_img[:, transform_dist[0] : w1 + transform_dist[0]], result_img[:, w1 + transform_dist[0] :])
    return result_img
# blend

# method used to find sift descriptors, keypoints and get the best 500 matches between 2 images
# start:
def find_matches(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matches = knnMatch(des1, des2)
    return matches, kp1, kp2
# end

# method used to stitch 2 images if enough matches are found, else img1 is returned
# start:
def stitch_images(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    good_matches, kp1, kp2 = find_matches(img1_gray, img2_gray)

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m['queryIdx']].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m['trainIdx']].pt for m in good_matches]).reshape(-1, 2)
        H, inliers = findHomography(src_pts, dst_pts)
        if len(inliers) > .3 * len(good_matches):
            res = stitch(img2, img1, H)
            return res
        else:
            print ("Not enough inliers found - %d/%d" % (len(inliers), len(good_matches)))
            return img1
    else:
        print ("Not enough matches found - %d/%d" % (len(good_matches), 10))
        return img1
# end

def main():
    folder = sys.argv[1]
    imgList = os.listdir(folder)
    imgs = []
    for img in imgList:
        if '.jpg' in img or '.jpeg' in img:
            imgs.append(img)
    print(imgs)
    img1 = cv2.imread(os.path.join(folder, imgs[0]))
    # down-sizing the image to 50% of its size to enhance performance
    size1 = (int(img1.shape[1] * 0.5), int(img1.shape[0] * 0.5))
    img1 = cv2.resize(img1, size1)

    for index in range(1, len(imgs)):
        img2 = cv2.imread(os.path.join(folder, imgs[index]))
        size2 = (int(img2.shape[1] * 0.5), int(img2.shape[0] * 0.5))
        img2 = cv2.resize(img2, size2)
        img1 = trim(stitch_images(img1, img2))

    # up-sizing the image back to 200%
    size1 = (int(img1.shape[1] * 2), int(img1.shape[0] * 2))
    img1 = cv2.resize(img1, size1)

    cv2.imwrite(os.path.join(folder, "panorama.jpg"), img1)

if __name__ == '__main__':
    main()
