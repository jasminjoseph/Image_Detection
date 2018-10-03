import cv2
import numpy

img1 = cv2.imread("book.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("tiltedBook.jpg", cv2.IMREAD_GRAYSCALE)

orb =cv2.ORB_create()
keypoints1, descriptor1 = orb.detectAndCompute(img1, None)
keypoints2, descriptor2 = orb.detectAndCompute(img2, None)

#Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
match = bf.match(descriptor1, descriptor2)
match = sorted(match, key = lambda x:x.distance)

matchResult = cv2.drawMatches(img1, keypoints1, img2, keypoints2, match[:20], None, flags = 2)

cv2.imshow("Book", img1)
cv2.imshow("Tilted Book", img2)
cv2.imshow("Result", matchResult)

cv2.waitKey(0)
cv2.destroyAllWindows()

