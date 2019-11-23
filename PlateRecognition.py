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

# car image -> grayscale image -> binary image

car_image = imread(filename, as_gray=True)
print(car_image.shape)

gray_car_image = car_image * 255
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(gray_car_image, cmap="gray")
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value


# this gets all the connected regions and groups them together
label_image = measure.label(binary_car_image)

# getting the maximum width, height and minimum width and height that a license plate can be
plate_dimensions = (0.04*label_image.shape[0], 0.5*label_image.shape[0], 0.2*label_image.shape[1], 0.6*label_image.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions
plate_objects_cordinates = []

min_height, max_height, min_width, max_width = plate_dimensions
plate_objects_cordinates = []

fig, (ax1) = plt.subplots(1)
ax1.imshow(gray_car_image, cmap="gray")

# regionprops creates a list of properties of all the labelled regions
for region in regionprops(label_image):
    if region.area < 50:
        #if the region is so small then it's likely not a license plate
        continue
        # the bounding box coordinates
    min_row, min_col, max_row, max_col = region.bbox

    region_height = max_row - min_row
    region_width = max_col - min_col

    # ensuring that the region identified satisfies the condition of a typical license plate
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:

        plate_like_objects.append(binary_car_image[min_row:max_row,
                                    min_col:max_col])
        plate_objects_cordinates.append((min_row, min_col,
                                            max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
                                        linewidth=2, fill=False)
        ax1.add_patch(rectBorder)
        Cropped = gray_car_image[min_row:max_row, min_col:max_col]
        
        text = pytesseract.image_to_string(Cropped, config='--psm 11')
        # print("Predicted Number by pytessaract : ",text)
        # break
        # let's draw a red rectangle over those regions
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

list_of_plates = []
list_of_columns = []
for lp in plate_like_objects:
    license_plate = np.invert(lp)

    labelled_plate = measure.label(license_plate)

    fig, ax1 = plt.subplots(1)
    license_plate = rgb2gray(license_plate)
    ax1.imshow(license_plate, cmap="gray")

    character_dimensions = (0.3*license_plate.shape[0], 1.0*license_plate.shape[0], 0.02*license_plate.shape[1], 0.6*license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions

    characters = []
    column_list = []

    rois = []

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
            rois.append(roi)

            # resize the characters to 20X20 and then append each character into the characters list
            resized_char = Image.fromarray(roi).resize((20, 20))
            
            characters.append(resized_char)

            # this is just to keep track of the arrangement of the characters
            column_list.append(x0)
    list_of_plates.append(characters)
    list_of_columns.append(column_list)
    plt.show()

for i in range(len(list_of_plates)):
    characters = list_of_plates[i]
    plate_num = []
    for resized_char in characters:
        roi = np.array(resized_char)
        # print(roi)
        roi = roi.reshape((1,400))
        valActivations  = nn1.feedforward(roi)
        pred = np.argmax(valActivations[-1], axis=1)
        
        if(valActivations[-1][0][pred]<0.5):
            plate_num.append('')
            continue

        if(pred<10):
            # print(pred[0])
            plate_num.append(str(pred[0]))
        else:
            # print(chr(65+pred-10))
            plate_num.append(str(chr(65+pred[0]-10)))

    column = np.array(list_of_columns[i])
    sort_idx = np.argsort(column)
    plate_num = np.array(plate_num)[sort_idx]
    print("".join(plate_num))