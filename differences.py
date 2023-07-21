from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def image_diff(image_path1, image_path2, output_path=None):
    # Open the images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Convert the images to numpy arrays
    array1 = np.array(image1)
    array2 = np.array(image2)

    # Calculate the absolute difference between the images
    diff = np.abs(array1 - array2)

    if(diff.any()>50):
        # Create a black and white image highlighting the differences
        diff_image = Image.fromarray(diff.astype('uint8'))
        # Display the images using matplotlib
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image1)
        plt.title('Image 1')

        plt.subplot(1, 3, 2)
        plt.imshow(image2)
        plt.title('Image 2')

        plt.subplot(1, 3, 3)
        plt.imshow(diff_image)
        plt.title('Differences')

        plt.show()

        # Save the difference image if output_path is provided
        if output_path:
            diff_image.save(output_path)
    else:


# Usage example:
image1_path = "C://aditi/competitions/hackrxGit/bajajss1.jfif"
image2_path = "C://aditi/competitions/hackrxGit/bajajss2.jfif"
output_diff_path = 'C://aditi/competitions/hackrxGit/diff1.jfif'

image_diff(image1_path, image2_path, output_diff_path)

""" 
import cv2 as cv
import numpy as np
from PIL import Image, ImageChops

#Ideal Image and The main Image
img2= cv.imread("C://aditi/competitions/hackrxGit/bajajss1.jfif")
img1 = cv.imread("C://aditi/competitions/hackrxGit/bajajss2.jfif")


#Verifys if there is or isnt a differance in the Image for the If statement
diff = cv.subtract(img2, img1)
diff1=np.array(diff)
results = not np.any(diff1)

#Tells the User if there is a Differance within the 2 images with the model image and the image given
if results is True:
    print("The Images are the same!")

else:
    print("The images are differant")


#This is to make the image show the differance to circle
img_1=Image.open(img1)
img_2=Image.open(img2)
diff1=ImageChops.difference(img_1,img_2)
diff1.save("Differance.jpg")

#Reads the image Just saved
Differance = cv.imread("Differance.jpg", 0)

#Resize the Image to make it smaller

img1s = cv.resize(img1, (0, 0), fx=0.5, fy=0.5)
Differance = cv.resize(Differance, (0, 0), fx=0.5, fy=0.5)    

# Find anything not black, i.e. The differance
nz = cv.findNonZero(Differance)

# Find top, bottom, left and right edge of the Differance
a = nz[:,0,0].min()
b = nz[:,0,0].max()
c = nz[:,0,1].min()
d = nz[:,0,1].max()

# Average top and bottom edges, left and right edges, to give centre
c0 = (a+b)/2
c1 = (c+d)/2

#The Center Coords
c3 = (int(c0),int(c1))

#Values for the below code so it doesnt look messy
radius = 50
color = (0, 0, 255)
thickness = 2

#This Places a Circle around the center of the differance
Finished = cv.circle(img1s, c3, radius, color, thickness)

#Saves the Final Image with the circle around it
cv.imwrite("Final.jpg", Finished) """