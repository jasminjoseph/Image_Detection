import numpy as np
import cv2

img1 = cv2.imread("stopOrig.jpeg",0)          # queryImage
img2 = cv2.imread("stopRot.jpg",0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# Brute Force matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Finding best matches
good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append(m)

print (len(good))

#Homograph
if len(good) > 10:
	query_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
	train_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
	#print(train_pts)

	matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
	matches_mask = mask.ravel().tolist()

	h, w = img1.shape
	pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(pts, matrix)

	homography = cv2.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 3)
	cv2.imshow("Homography", homography)
else:
	cv2.imshow("Homography", img2)


#cv2.imshow("result", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

