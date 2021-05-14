import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt

img = cv2.imread("image.png")
# 1. task
gray_scale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imwrite("Grayscale.png",gray_scale)

# 2.
def flipImage (image, flag):

    flipped_image = copy.deepcopy(image)
    height, width = flipped_image.shape

    index1 = 0
    index2 = 0
    if flag == 'v': #vertical flip
        for row in range(height):
            for col in range(width // 2):
                temp= image[row][col]
                flipped_image[row][col] = flipped_image[row][width - col - 1]
                flipped_image[row][width - col - 1] = temp
        cv2.imwrite("Vertical_flip.png", flipped_image)


    if flag == 'h':  # horizontal flip
        for row in range(height // 2):
            for col in range(width):
                temp = flipped_image[row][col]
                flipped_image[row][col] = flipped_image[height - row -1][col]
                flipped_image[height - row -1][col] = temp
        cv2.imwrite("Horizontal_flip.png", flipped_image)


# get range
dataType = gray_scale.dtype
min_value = np.iinfo(dataType).min
max_value = np.iinfo(dataType).max

intensityRange = max_value + 1

# 3.
def generateHistogram (image, intensityRange ):

    histogram = np.zeros(intensityRange )
    for index1 in image:
        for index2 in index1:
            histogram[index2] += 1

    rangeValues = np.zeros(intensityRange)
    for index in range(intensityRange):
        rangeValues[index] = index

    plt.figure()
    plt.plot(rangeValues ,histogram)
    plt.fill_between(rangeValues ,histogram)
    plt.show()

    return histogram

# 4.
def equalizeHistogram(image, histogram, intensityRange, totalPixels ):

    #intensityRange -1 # L -1
    processedIntensity = np.zeros(intensityRange, dtype = int) # s(k) array
    totalSum = 0

    #find s(k) new values
    for index in range(intensityRange):

        processedIntensityItem = 0 # s(k)
        totalSum += histogram[index] / totalPixels

        processedIntensityItem = totalSum * (intensityRange - 1)
        processedIntensityItem = round(processedIntensityItem)

        processedIntensity[index] = processedIntensityItem

    # find ps(sk)
    '''ps_sk = np.zeros(intensityRange)

    for index in np.unique(processedIntensity):
        newPixelCount = 0

        oldVals = np.where(processedIntensity == index)

        for element1 in oldVals:
            for element2 in element1:
                newPixelCount += histogram[element2]

        # divide to total pixel number
        ps_sk[index] = newPixelCount / totalPixels
        # now ps_sk - sk graph is equalized histogram
    '''

    # render equalized image
    index1 = 0
    index2 = 0
    for row in image:
        for element in row:
            image[index1][index2] = processedIntensity[element]
            index2 += 1
        index2 = 0
        index1 += 1

    cv2.imwrite("Equalized_image.png",image)

    # plot histogram
    generateHistogram(image, intensityRange)


height, width = gray_scale.shape
totalPixels = height * width  # n

flipImage(gray_scale, 'v')
flipImage(gray_scale, 'h')

histogram = generateHistogram(gray_scale, intensityRange)
equalizeHistogram(gray_scale, histogram,intensityRange , totalPixels)