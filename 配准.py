import cv2
import numpy as np

# 读取图像
image1 = cv2.imread('/home/pxl/myProject/血管分割/ImgFA02_PRIME-FP20.tif', 0)  # 读取原始图像
image2 = cv2.imread('/home/pxl/myProject/血管分割/LabelFP02_PRIME-FP20.png', 0)  # 读取分割图

# 使用SIFT检测特征点并计算描述符
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# 使用FLANN匹配特征点
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 选择优秀的匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 获取匹配点的位置
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

# 计算变换矩阵
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 应用变换矩阵对图像进行配准
height, width = image1.shape
aligned_image = cv2.warpPerspective(image1, M, (width, height))

# 显示结果
# 保存配准后的图像
cv2.imwrite('aligned_image.png', aligned_image)