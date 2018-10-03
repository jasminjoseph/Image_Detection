import cv2
import numpy as np

img1 = cv2.imread("stop.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("stop_sign.jpg", cv2.IMREAD_GRAYSCALE)

orb =cv2.ORB_create()
kp1, desc1 = orb.detectAndCompute(img1, None)
kp2, desc2 = orb.detectAndCompute(img2, None)

# Feature matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc1, desc2)
matches = sorted(matches, key = lambda x:x.distance)
 
match_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)


# Homography
if len(matches[:20]) > 10:
	query_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        train_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
 
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
 
        # Perspective transform
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
 
        homography = cv2.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 3)
 
        cv2.imshow("Homography", homography)
else:
	cv2.imshow("Homography", grayframe)

key = cv2.waitKey(1)
    


#cv2.imshow("Book", img1)
#cv2.imshow("Tilted Book", img2)
#cv2.imshow("Result", match_result)

#cv2.waitKey(0)
cv2.destroyAllWindows()

