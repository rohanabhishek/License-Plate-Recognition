import sys
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
from skimage.color import rgb2gray

from skimage.io import imread
from skimage.filters import threshold_otsu
import pytesseract
from PIL import Image
import imutils

from tasks import *

plate_like_objects = []
filename = sys.argv[1]

# Image converted to binary
car_image = imread(filename, as_gray=True)
print(car_image.shape)

gray_car_image = car_image*255
# thershold value obtained using Otsu's method
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value

# get all the connected regions and group them together
label_image = measure.label(binary_car_image)

# constraints on maximum and minimum values on width, height
plate_dimensions = (0.04*label_image.shape[0], 0.5*label_image.shape[0], 0.2*label_image.shape[1], 0.6*label_image.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions

plate_objects_cordinates = []

fig, (ax1) = plt.subplots(1)
ax1.imshow(gray_car_image, cmap="gray")

# regionprops creates a list of properties of all the labelled regions
for region in regionprops(label_image):
    if region.area < 50:
        #if the region is very small
        continue
    # the bounding box coordinates
    min_row, min_col, max_row, max_col = region.bbox

    region_height = max_row - min_row
    region_width = max_col - min_col

    # checking the conditions of a typical license plate
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:

        plate_like_objects.append(gray_car_image[min_row:max_row,
                                    min_col:max_col])
        plate_objects_cordinates.append((min_row, min_col,
                                            max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
                                        linewidth=2, fill=False)
        # red rectangular border added
        ax1.add_patch(rectBorder)
        Cropped = gray_car_image[min_row:max_row, min_col:max_col]
        
        # text = pytesseract.image_to_string(Cropped, config='--psm 11')
        # print("Predicted Number by pytessaract : ",text)
plt.show()

modelName = 'my_model.npy'
nn1 = nn.NeuralNetwork(36, 0.001, 200, 10)
nn1.addLayer(FullyConnectedLayer(400, 50, "relu"))
nn1.addLayer(FullyConnectedLayer(50, 36, "softmax"))

model = np.load(modelName,allow_pickle=True)
k,i = 0,0
for l in nn1.layers:
    if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer": 
        nn1.layers[i].weights = model[k]
        nn1.layers[i].biases = model[k+1]
        k+=2
    i+=1
print("Model Loaded... ")

list_of_plates = [] # list of characters in all paltes
list_of_columns = [] # to re-order characters as they are in LP
for lp in plate_like_objects:
    # invert image
    license_plate = (255-lp)
    # reaply threshold on the extracted region
    threshold_value = threshold_otsu(license_plate)
    license_plate = license_plate > threshold_value

    labelled_plate = measure.label(license_plate)

    fig, ax1 = plt.subplots(1)
    license_plate = rgb2gray(license_plate)
    ax1.imshow(license_plate, cmap="gray")

    # character dimension constraints
    character_dimensions = (0.3*license_plate.shape[0], 1.0*license_plate.shape[0], 0.01*license_plate.shape[1], 0.6*license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions

    characters = []
    column_list = []

    for regions in regionprops(labelled_plate):

        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0

        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roi = license_plate[y0:y1, x0:x1]

            # draw a red bordered rectangle over the character.
            rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                        linewidth=2, fill=False)
            ax1.add_patch(rect_border)

            # resize the characters to 20X20 and then append each character into the characters list
            resized_char = Image.fromarray(roi).resize((20, 20))
            characters.append(resized_char)

            # to keep track of the arrangement of the characters(based on x-coordinate)
            column_list.append(x0)
    list_of_plates.append(characters)
    list_of_columns.append(column_list)
    plt.show()

list_of_numbers = []
for i in range(len(list_of_plates)):
    characters = list_of_plates[i]
    plate_num = []
    for resized_char in characters:
        roi = np.array(resized_char)

        # reshape to an array as one input
        roi = roi.reshape((1,400))

        # predict result using neural network
        valActivations  = nn1.feedforward(roi)

        # get the class with highest prediction
        pred = np.argmax(valActivations[-1], axis=1)
        
        # check with threshold to remove non-characters
        if(valActivations[-1][0][pred]<0.5):
            plate_num.append('')
            continue

        if(pred<10):
            plate_num.append(str(pred[0]))
        else:
            plate_num.append(str(chr(65+pred[0]-10)))

    column = np.array(list_of_columns[i])

    # sort characters as they are in LP
    sort_idx = np.argsort(column)
    plate_num = np.array(plate_num)[sort_idx]

    # output licence plate number
    plate_num = "".join(plate_num)
    list_of_numbers.append(plate_num)

print('Predictions - ',end=' ')
print(list_of_numbers)

final_num = sorted(list_of_numbers, key=len) 
print('Final Licence plate - ' + final_num[-1])