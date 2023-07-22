#k-means clustering for colors 

""" # import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils 
import cv2
# construct the argument parser and parse the arguments

# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread("C://aditi/competitions/hackerx4/contrastDataset/train/badContrast/001-Bad-Web-Design.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img1 = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)

# show our image
plt.figure()
plt.axis("off")
plt.imshow(img1)
plt.show() #nescessary to display above image.

# reshape the image to be a list of pixels
img2 = img1.reshape((img1.shape[0] * img1.shape[1], 3))
# cluster the pixel intensities
clt = KMeans(n_clusters = 10)
clt.fit(img2)

# import the necessary packages
import numpy as np
import cv2
def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(img2)) + 1)
	(hist, _) = np.histogram(img2, bins = numLabels) #clt.labels_
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	# return the histogram
	return hist

def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	
	# return the bar chart
	return bar

# build a histogram of clusters and then create a figure (utils.)
# representing the number of pixels labeled to each color
hist = centroid_histogram(img2)
#bar = plot_colors(hist, clt.cluster_centers_)
plt.plot(hist)
plt.show()
# show our color bart
""" 
"""plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()

 """

def find_number_of_colors(imgPath):
    from PIL import Image
    import numpy as np
    #imgPath = 'C://aditi/competitions/hackerx4/contrastDataset/train/badContrast/3.jpg'

    img = Image.open(imgPath)
    img = img.resize((224, 224))
    uniqueColors = set()

    w, h = img.size
    for x in range(w):
        for y in range(h):
            pixel = img.getpixel((x, y))
            uniqueColors1=np.array(list(uniqueColors))
            # if(x==100 & y==100):
            #     print(pixel[1])
            flag=1
            for val in uniqueColors1:
                dist = np.linalg.norm(pixel-val)
                if dist<100:
                    flag=0
                    break
                #print(val)
            if flag==1:
                uniqueColors.add(pixel)

    totalUniqueColors = len(uniqueColors)
    # uniqueColors1=np.array(list(uniqueColors))
    # print(uniqueColors1[1])
    print(totalUniqueColors)
    return(totalUniqueColors)