""" # this is sample file
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

""" 
#img1 = img1.resize((256, 256))
#img2 = img2.resize((256, 256)) 
"""

# applying grayscale method
img11= ImageOps.grayscale(img1)
 
img11.show()

img22=ImageOps.grayscale(img2)
 
img22.show()

# finding difference
diff = ImageChops.difference(img11, img22)
  
# showing the difference
diff.show() """


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
cv2.imwrite('C://aditi/competitions/hackrxGit/aligned.jpg', transformed_img)
  
# assign images
img1 = Image.open("C://aditi/competitions/hackrxGit/bajajss1.jfif")
img2 = Image.open("C://aditi/competitions/hackrxGit/aligned.jpg")

""" img1 = img1.resize((256, 256))
img2 = img2.resize((256, 256)) """

# applying grayscale method
img11= ImageOps.grayscale(img1)
 
#img11.show()

img22=ImageOps.grayscale(img2)
 
#img22.show()

# finding difference
diff = ImageChops.difference(img11, img22)
  
# showing the difference
#diff.show()


import cv2 as cv
import numpy as np

# load images
img1 = cv.imread("C://aditi/competitions/hackrxGit/bajajss1.jfif")
img2 = cv.imread("C://aditi/competitions/hackrxGit/aligned.jpg")

# calculate difference
diff = cv.subtract(img1, img2)  # other order `(img2, img1)` gives worse result

# saves difference
cv.imwrite("C://aditi/competitions/hackrxGit/difference.png", diff)

# show difference - press any key to close
# cv.imshow('diff', diff)
# cv.waitKey(0)
# cv.destroyWindow('diff')

if not np.any(diff):
    print("The images are the same!")
else:
    print("The images are different")

# resize images to make them smaller
# img1_resized = cv.resize(img1, (224, 224), interpolation=cv2.INTER_NEAREST)
# diff_resized = cv.resize(diff, (224, 224), interpolation=cv2.INTER_NEAREST)    #(img1, (0, 0), fx=0.5, fy=0.5)
img1_resized = img1
diff_resized = diff

# convert to grayscale (without saving and loading again)
diff_resized = cv.cvtColor(diff_resized, cv.COLOR_BGR2GRAY)
ret,thresh1 = cv.threshold(diff_resized,127,255,cv.THRESH_BINARY)

# find anything not black in differance
non_zero = cv.findNonZero(thresh1)
#print(non_zero)
cv.imshow('binary thresholded image', thresh1)
cv.imwrite("C://aditi/competitions/hackrxGit/binarythresholdedimage.jpg", thresh1)
          
# find top, bottom, left and right edge of the differance
x_min = non_zero[:,0,0].min()
x_max = non_zero[:,0,0].max()
y_min = non_zero[:,0,1].min()
y_max = non_zero[:,0,1].max()
print('x:', x_min, x_max)
print('y:', y_min, y_max)

sizes = [x_max-x_min+1, y_max-y_min+1]
print('width :', sizes[0])
print('height:', sizes[1])

# center 
center_x = (x_min + x_max) // 2
center_y = (y_min + y_max) // 2
center = (center_x, center_y)
print('center:', center)

# radius 
radius = max(sizes) // 2
print('radius:', radius)

color = (0, 0, 255)
thickness = 2

start_point=(x_min, y_min)
end_point=(x_max, y_max)

# draw circle around the center of the differance
finished = cv2.rectangle(img2, start_point, end_point, color, thickness)
#finished = cv.circle(img2, center, radius, color, thickness)

# saves final image with circle
cv.imwrite("C://aditi/competitions/hackrxGit/final.png", finished)

# show final image - press any key to close
cv.imshow('finished', finished)
cv.waitKey(0)