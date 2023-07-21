# this is sample file
# import module
from PIL import Image, ImageChops, ImageOps

import cv2
import numpy as np
  
# Open the image files.
img1_color = cv2.imread("C://aditi/competitions/hackrxGit/bajajss1.jfif")  # Image to be aligned.
img2_color = cv2.imread("C://aditi/competitions/hackrxGit/bajajss2.jfif")    # Reference image.
  
# Convert to grayscale.
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
height, width = img2.shape
  
# Create ORB detector with 5000 features.
orb_detector = cv2.ORB_create(5000)
  
# Find keypoints and descriptors.
# The first arg is the image, second arg is the mask
#  (which is not required in this case).
kp1, d1 = orb_detector.detectAndCompute(img1, None)
kp2, d2 = orb_detector.detectAndCompute(img2, None)
  
# Match features between the two images.
# We create a Brute Force matcher with 
# Hamming distance as measurement mode.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
  
# Match the two sets of descriptors.
matches = matcher.match(d1, d2)

#matches=np.array(matches)
  
# Sort matches on the basis of their Hamming distance.
#matches.sort(key = lambda x: x.distance)
matches = tuple(sorted(matches, key=lambda x: x.distance))
  
# Take the top 90 % matches forward.
matches = matches[:int(len(matches)*0.9)]
no_of_matches = len(matches)
  
# Define empty matrices of shape no_of_matches * 2.
p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))
  
for i in range(len(matches)):
  p1[i, :] = kp1[matches[i].queryIdx].pt
  p2[i, :] = kp2[matches[i].trainIdx].pt
  
# Find the homography matrix.
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
  
# Use this matrix to transform the
# colored image wrt the reference image.
transformed_img = cv2.warpPerspective(img1_color,
                    homography, (width, height))
  
# Save the output.
cv2.imwrite('C://aditi/competitions/hackrxGit/align.jpg', transformed_img)
  
# assign images
img1 = Image.open("C://aditi/competitions/hackrxGit/bajajss1.jfif")
img2 = Image.open("C://aditi/competitions/hackrxGit/align.jpg")

""" img1 = img1.resize((256, 256))
img2 = img2.resize((256, 256)) """

# applying grayscale method
img11= ImageOps.grayscale(img1)
 
img11.show()

img22=ImageOps.grayscale(img2)
 
img22.show()

# finding difference
diff = ImageChops.difference(img11, img22)
  
# showing the difference
diff.show()