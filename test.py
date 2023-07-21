""" import cv2 as cv
import numpy as np
from PIL import Image, ImageChops
import matplotlib.pyplot as plt

# Set the path to the images
img1_path = "C://aditi/competitions/hackrxGit/bajajss1.jfif"
img2_path = "C://aditi/competitions/hackrxGit/bajajss2.jfif"

# Load the images
img1 = cv.imread(img1_path)
img2 = cv.imread(img2_path)

# Calculate the difference between the two images
diff = cv.subtract(img2, img1)

# Check if there are any non-zero values in the difference image
results = not np.any(diff)

# Print a message indicating whether the images are the same or different
if results:
    print("The Images are the same!")
else:
    print("The images are different")

# Load the images using PIL
img_1 = Image.open(img1_path)
img_2 = Image.open(img2_path)

# Calculate the difference between the two images using PIL
diff = ImageChops.difference(img_1, img_2)

# Save the difference image
diff.save("Difference.jpg")

# Load the difference image using OpenCV
Difference = cv.imread("Difference.jpg", 0)

# Resize the images to make them smaller
img1s = cv.resize(img1, (0, 0), fx=0.5, fy=0.5)
Difference = cv.resize(Difference, (0, 0), fx=0.5, fy=0.5)

# Find anything not black in the difference image
nz = cv.findNonZero(Difference)

if nz is not None:
    # Find top, bottom, left and right edge of the difference
    a = nz[:, 0, 0].min()
    b = nz[:, 0, 0].max()
    c = nz[:, 0, 1].min()
    d = nz[:, 0, 1].max()

    # Average top and bottom edges, left and right edges, to give center
    c0 = (a + b) / 2
    c1 = (c + d) / 2

    # The center coordinates
    c3 = (int(c0), int(c1))

    # Values for drawing a circle around the center of the difference
    radius = 50
    color = (0, 0, 255)
    thickness = 2

    # Draw a circle around the center of the difference
    Finished = cv.circle(img1s, c3, radius, color, thickness)

    # Save the final image with the circle around it
    cv.imwrite("Final.jpg", Finished)
else:
    print("No differences found") """


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# load images
img1 = cv.imread("C://aditi/competitions/hackrxGit/bajajss1.jfif")
img2 = cv.imread("C://aditi/competitions/hackrxGit/bajajss2.jfif")

# calculate difference
diff = cv.subtract(img1, img2)  # other order `(img2, img1)` gives worse result

# saves difference
cv.imwrite("difference.png", diff)

# show difference - press any key to close
cv.imshow('diff', diff)
cv.waitKey(0)
#cv.destroyWindow('diff')

if not np.any(diff):
    print("The images are the same!")
else:
    print("The images are differant")

# resize images to make them smaller
#img1_resized = cv.resize(img1, (224, 224), interpolation=cv.INTER_NEAREST)
#diff_resized = cv.resize(diff, (224, 224), interpolation=cv.INTER_NEAREST)    #(img1, (0, 0), fx=0.5, fy=0.5)
img1_resized = img1
diff_resized = diff

# convert to grayscale (without saving and loading again)
diff_resized = cv.cvtColor(diff_resized, cv.COLOR_BGR2GRAY)

# find anything not black in differance
non_zero = cv.findNonZero(diff_resized)
#print(non_zero)

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
rectangle_width = x_max - x_min
rectangle_height = y_max - y_min

color = (0, 0, 255)
thickness = 2

# draw circle around the center of the differance
#finished = cv.rectangle(img1_resized, center, radius, color, thickness)

# Define the top-left and bottom-right corners of the rectangle
top_left = (center_x - rectangle_width // 2, center_y - rectangle_height // 2)
bottom_right = (center_x + rectangle_width // 2, center_y + rectangle_height // 2)

# Draw the rectangle around the center of the difference
cv.rectangle(img1_resized, top_left, bottom_right, color, thickness)

# Save the final image with the rectangle
cv.imwrite("final.png", img1_resized)
""" 
# Show the final image with the rectangle - press any key to close
cv.imshow('finished', img1_resized)
cv.waitKey(0)
cv.destroyAllWindows()
 """
img_rgb = cv.cvtColor(img1_resized, cv.COLOR_BGR2RGB)

# Show the image using Matplotlib
plt.imshow(img_rgb)
plt.axis('off')  # Remove axes ticks and labels
plt.show()



""" # saves final image with circle
cv.imwrite("final.png", finished)

# show final image - press any key to close

cv.imshow('finished', finished)
cv.waitKey(0)
cv.destroyWindow('finished') """